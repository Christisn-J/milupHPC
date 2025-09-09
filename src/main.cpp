#include "../include/miluphpc.h"
#include "../include/integrator/explicit_euler.h"
#include "../include/integrator/predictor_corrector_euler.h"
#include "../include/integrator/leapfrog.h"
#include "../include/utils/config_parser.h"
#include "../include/utils/compile_checks.h"
#include "../include/constants.h"
#include <boost/filesystem.hpp>
#include <boost/algorithm/string.hpp>

#include <fenv.h>
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <string.h>
#include <random>

#define ENV_LOCAL_RANK "OMPI_COMM_WORLD_LOCAL_RANK"

std::string get_path_OutputDirectory(const std::string& configDir,
                               const std::string& parserDir,
                               const std::string& inputFile)
{
    // 1. Prefer parser value if valid
    if (!parserDir.empty() && parserDir != "-") {
        Logger(DEBUG) << "Using output directory from parser: " << parserDir;
        return parserDir;
    }

    // 2. Trim config directory
    std::string trimmed = boost::algorithm::trim_copy(configDir);

    // 3. Detect placeholder-like config entries
    std::regex placeholderPattern(R"(<\s*(todo|to be set|placeholder)[^>]*>?)", std::regex_constants::icase);
    if (trimmed.empty() || std::regex_search(trimmed, placeholderPattern)) {
        // 4. Generate fallback: "./output/<basename of input file>"
        boost::filesystem::path inputPath(inputFile);
        std::string baseName = inputPath.stem().string();
        std::string fallback = "./output/" + baseName;
        Logger(INFO) << "No valid output directory specified. Using fallback: " << fallback;
        return fallback;
    }

    Logger(DEBUG) << "Using output directory from config: " << trimmed;
    return trimmed;
}


void createOutputDirectory(const std::string& dir, std::string& logDirOut)
{
    namespace fs = boost::filesystem;

    if (!fs::exists(dir)) {
        if (fs::create_directories(dir)) {
            Logger(DEBUG) << "Created output directory: " << dir;
        } else {
            Logger(ERROR) << "Failed to create output directory: " << dir;
            MPI_Finalize();
            exit(1);
        }
    } else if (!fs::is_directory(dir)) {
        Logger(ERROR) << "Output path exists but is not a directory: " << dir;
        MPI_Finalize();
        exit(1);
    }

    logDirOut = dir + "/log/";

    if (!fs::exists(logDirOut)) {
        if (fs::create_directories(logDirOut)) {
            Logger(DEBUG) << "Created log directory: " << logDirOut;
        } else {
            Logger(ERROR) << "Failed to create log directory: " << logDirOut;
            MPI_Finalize();
            exit(1);
        }
    } else if (!fs::is_directory(logDirOut)) {
        Logger(ERROR) << "Log path exists but is not a directory: " << logDirOut;
        MPI_Finalize();
        exit(1);
    }
}


void SetDeviceBeforeInit()
{
    char * localRankStr = NULL;
    int rank = 0;
    int devCount = 2;

    if ((localRankStr = getenv(ENV_LOCAL_RANK)) != NULL)
    {
        rank = atoi(localRankStr);
    }

    gpuErrorcheck(cudaGetDeviceCount(&devCount));
    //gpuErrorcheck(cudaSetDevice(rank % devCount));
    gpuErrorcheck(cudaSetDevice(rank % devCount));
}

structLog LOGCFG = {};

int main(int argc, char** argv)
{

    /// SETTINGS/INITIALIZATIONS
    // -----------------------------------------------------------------------------------------------------------------

    /// MPI rank setting
    SetDeviceBeforeInit();

    boost::mpi::environment env(argc, argv);
    boost::mpi::communicator comm;

    int rank = comm.rank();
    int numProcesses = comm.size();

    /// Setting CUDA device
    //cudaSetDevice(rank);
    int device;
    cudaGetDevice(&device);
    int numDevices;
    cudaGetDeviceCount(&numDevices);
    cudaDeviceSynchronize();

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, device);
    int numberOfMultiprocessors = deviceProp.multiProcessorCount;
    printf("numberOfMultiprocessors: %i\n", numberOfMultiprocessors);

    // testing whether MPI works ...
    //int mpi_test = rank;
    //all_reduce(comm, boost::mpi::inplace_t<int*>(&mpi_test), 1, std::plus<int>());
    //Logger(INFO) << "mpi_test = " << mpi_test;

    /// Command line argument parsing
    cxxopts::Options options("HPC NBody", "Multi-GPU CUDA Barnes-Hut NBody/SPH code");

    bool loadBalancing = Default::loadBalancing;

    options.add_options()
            ("n,number-output-files", "number of output files", cxxopts::value<int>()->default_value(std::to_string(Default::numberFiles)))
            ("t,max-time-step", "time step", cxxopts::value<real>()->default_value(DefaultValue<real>::str()))
            ("l,load-balancing", "load balancing", cxxopts::value<bool>(loadBalancing))
            ("L,load-balancing-interval", "load balancing interval", cxxopts::value<int>()->default_value(DefaultValue<integer>::str()))
            ("C,config", "config file", cxxopts::value<std::string>()->default_value("config/config.info"))
            ("m,material-config", "material config file", cxxopts::value<std::string>()->default_value("config/material.cfg"))
            ("c,curve-type", "curve type (Lebesgue: 0/Hilbert: 1)", cxxopts::value<int>()->default_value(DefaultValue<integer>::str()))
            ("f,input-file", "File name", cxxopts::value<std::string>()->default_value(DefaultValue<std::string>::str()))
			("o,output", "Output directory", cxxopts::value<std::string>()->default_value(DefaultValue<std::string>::str()))
            ("v,verbosity", "Verbosity level", cxxopts::value<int>()->default_value(std::to_string(Default::verbose_lvl)))
            ("h,help", "Show this help");

    cxxopts::ParseResult result;
    try {
        result = options.parse(argc, argv);
    }
    catch (cxxopts::OptionException &exception) {
        std::cerr << exception.what();
        MPI_Finalize();
        exit(0);
    }
    catch (...) {
        std::cerr << "Error parsing ...";
        MPI_Finalize();
        exit(0);
    }

    if (result.count("help")) {
        if (rank == 0) {
            std::cout << options.help() << std::endl;
        }
        MPI_Finalize();
        exit(0);
    }

    /// Config file parsing
    std::string configFile = result["config"].as<std::string>();
    checkFileAvailable(configFile, true, std::string{"Provided config file not available!"});

    /// Collect settings/information in struct
    SimulationParameters parameters;

    parameters.verbosity = result["verbosity"].as<int>();
    /// Logger settings
    LOGCFG.headers = true;
    if (parameters.verbosity == 0) {
        LOGCFG.level = TRACE;
    }
    else if (parameters.verbosity == 1) {
        LOGCFG.level = INFO;
    }
    else if (parameters.verbosity >= 2) {
        LOGCFG.level = DEBUG;
    }
    else {
        LOGCFG.level = TRACE;
    }
    //LOGCFG.level = static_cast<typeLog>(1); //TRACE; //DEBUG;
    LOGCFG.rank = rank;
    //LOGCFG.outputRank = 0;
    //Logger(DEBUG) << "DEBUG output";
    //Logger(WARN) << "WARN output";
    //Logger(ERROR) << "ERROR output";
    //Logger(INFO) << "INFO output";
    //Logger(TRACE) << "TRACE output";
    //Logger(TIME) << "TIME output";

    Logger(DEBUG) << "rank: " << rank << " | number of processes: " << numProcesses;
    Logger(DEBUG) << "device: " << device << " | num devices: " << numDevices;

    Logger(INFO) << "sizeof(keyType) = " << (int)sizeof(keyType); // intmax_t, uintmax_t
    Logger(INFO) << "sizeof(intmax_t) = " << (int)sizeof(intmax_t);
    Logger(INFO) << "sizeof(uintmax_t) = " << (int)sizeof(uintmax_t); // __int128
    Logger(INFO) << "sizeof(__int128) = " << (int)sizeof(__int128);

    ConfigParser confP{ConfigParser(configFile.c_str())}; //"config/config.info"
    LOGCFG.write2LogFile = confP.getVal<bool>("log");
    LOGCFG.omitTime = confP.getVal<bool>("omitTime");

    std::string configDir = confP.getVal<std::string>("directory");
	std::string parserDir = result["output"].as<std::string>();
	std::string inputFile = result["input-file"].as<std::string>();

    // --- Output directory ---
	// Step 1: Decide which directory to use
	std::string finalOutputDir = get_path_OutputDirectory(configDir, parserDir, inputFile);

	// Step 2: Check if it's valid
    checkDirectoryAvailable(finalOutputDir, true, "Output directory is invalid!");

	// Step 3: Create directory (and log subdirectory)
	std::string finalLogDir;
	createOutputDirectory(finalOutputDir, finalLogDir);

	// Step 4: Store in parameters
	parameters.directory = finalOutputDir;
	parameters.logDirectory = finalLogDir;

    std::stringstream logFileName;
    logFileName << parameters.logDirectory << "miluphpc.log";
    LOGCFG.logFileName = logFileName.str();
    Logger(TRACE) << "log file to: " << logFileName.str();

    //MPI_Finalize();
    //exit(0);

    // --- Time-related parameters ---
    parameters.timeEnd = checkMinValue(confP.getVal<real>("timeEnd"), 0.0, InvalidValue<real>::value(), "timeEnd", "config", true);
    parameters.timeStep = checkInRange(confP.getVal<real>("timeStep"), 0.0, parameters.timeEnd, InvalidValue<real>::value(), "timeStep", "config", true);

    // Override maxTimeStep via command line if set, else use config
    parameters.maxTimeStep = result["max-time-step"].as<real>();
    if (parameters.maxTimeStep < 0.0) {
        parameters.maxTimeStep = confP.getVal<real>("maxTimeStep");
    }
    parameters.maxTimeStep = checkInRange(parameters.maxTimeStep, 0.0, parameters.timeEnd, parameters.timeStep, "maxTimeStep");

    // --- Output rank ---
    parameters.outputRank = confP.getVal<int>("outputRank");
    parameters.outputRank = checkInRange(parameters.outputRank, 0, numProcesses - 1, DefaultValue<integer>::value(), "outputRank");
    LOGCFG.outputRank = parameters.outputRank;

    // --- SFC (Space-filling curve) ---
    parameters.sfcSelection = confP.getVal<int>("sfc");
    if (result["curve-type"].as<int>() != InvalidValue<integer>::value()) {
        parameters.sfcSelection = result["curve-type"].as<int>();
    }
    parameters.sfcSelection = checkInRange(parameters.sfcSelection, 0, 1, 0, "sfcSelection");

    // --- Integrator ---
    parameters.integratorSelection = confP.getVal<int>("integrator");
    parameters.integratorSelection = checkInRange(parameters.integratorSelection, 0, 2, 0, "integratorSelection");

    // --- Output files ---
    parameters.numOutputFiles = checkMinValue(result["number-output-files"].as<int>(), 1, Default::numberFiles, "number-output-files");

    // --- Load balancing interval ---
    parameters.loadBalancingInterval = confP.getVal<int>("loadBalancingInterval");
    int cliBalancingInterval = result["load-balancing-interval"].as<int>();
    if (cliBalancingInterval > 0) {
        parameters.loadBalancingInterval = cliBalancingInterval;
    }
    parameters.loadBalancingInterval = checkMinValue(parameters.loadBalancingInterval, 1, 10, "loadBalancingInterval");

    parameters.loadBalancingBins = checkMinValue(confP.getVal<int>("loadBalancingBins"), 1, 2000, "loadBalancingBins");

    // --- Verbosity ---
    parameters.verbosity = checkInRange(result["verbosity"].as<int>(), 0, 2, Default::verbose_lvl, "verbosity");

    // --- Material File ---
    parameters.materialConfigFile = result["material-config"].as<std::string>();
    if (!checkFileAvailable(parameters.materialConfigFile, false)) {
        parameters.materialConfigFile = std::string{"config/material.cfg"};
        checkFileAvailable(parameters.materialConfigFile, true, std::string{"Provided material config file and default (config/material.cfg) not available!"});
    }

    // --- Initial Condition File ---
    parameters.inputFile = result["input-file"].as<std::string>();
    checkFileAvailable(parameters.inputFile, true, "Provided input file not available!");

    // --- Particle memory contingent ---
    parameters.particleMemoryContingent = confP.getVal<real>("particleMemoryContingent");
    parameters.particleMemoryContingent = checkInRange(parameters.particleMemoryContingent, 0.0, 1.0, 1.0, "particleMemoryContingent");

    parameters.timeKernels = true;
    parameters.performanceLog = checkBoolValue(confP, "performanceLog", false, "performanceLog");
    parameters.particlesSent2H5 = checkBoolValue(confP, "particlesSent2H5", false, "particlesSent2H5");
    parameters.removeParticles = checkBoolValue(confP, "removeParticles" , false, "removeParticles");

    // Load balancing: kombinierte Auswertung aus config + CLI
    bool configLB = checkBoolValue(confP, "loadBalancing", Default::loadBalancing);
    bool cliLB    = checkBoolValue(result, "load-balancing", Default::loadBalancing);
    parameters.loadBalancing = configLB || cliLB;


//#if GRAVITY_SIM
    parameters.theta = checkMinValue(confP.getVal<real>("theta"), 0.0, 0.5, "theta");
    parameters.smoothing = checkMinValue(confP.getVal<real>("smoothing"), 0.0, 0.01, "smoothing");
    parameters.gravityForceVersion = checkInRange(confP.getVal<int>("gravityForceVersion"), 0, 4, 0, "gravityForceVersion");
//#endif

//#if SPH_SIM
    parameters.smoothingKernelSelection = checkInRange(confP.getVal<int>("smoothingKernel"), 0, 5, 0, "smoothingKernelSelection");
    parameters.sphFixedRadiusNNVersion = checkInRange(confP.getVal<int>("sphFixedRadiusNNVersion"), 0, 3, 0, "sphFixedRadiusNNVersion");
//#endif

    parameters.removeParticlesCriterion = checkInRange(confP.getVal<int>("removeParticlesCriterion"), 0, 1, 0, "removeParticlesCriterion");
    parameters.removeParticlesDimension = checkMinValue(confP.getVal<real>("removeParticlesDimension"), 0.0, 10.0, "removeParticlesDimension");


//TODO: apply those
    parameters.calculateCenterOfMass = checkBoolValue(confP, "calculateCenterOfMass", false, "calculateCenterOfMass");
    parameters.calculateAngularMomentum = checkBoolValue(confP, "calculateAngularMomentum", false, "calculateAngularMomentum");
    parameters.calculateEnergy = checkBoolValue(confP, "calculateEnergy", false, "calculateEnergy");


// + 1 should not be necessary, but need to investigate whether this is a problem for 1 GPU sims
    parameters.domainListSize = POW_DIM * MAX_LEVEL * (numProcesses - 1) + 1;
    Logger(DEBUG) << "domainListSize: " << parameters.domainListSize;


    /// H5 profiling/profiler
    // profiling
    std::stringstream profilerFile;
    profilerFile << parameters.logDirectory << "performance.h5";
    H5Profiler &profiler = H5Profiler::getInstance(profilerFile.str());
    profiler.setRank(comm.rank());
    profiler.setNumProcs(comm.size());
    if (!parameters.performanceLog) {
        profiler.disableWrite();
    }
    // General
    profiler.createValueDataSet<int>(ProfilerIds::numParticles, 1);
    profiler.createValueDataSet<int>(ProfilerIds::numParticlesLocal, 1);
    profiler.createVectorDataSet<keyType>(ProfilerIds::ranges, 1, comm.size() + 1);
    // Timing
    profiler.createValueDataSet<real>(ProfilerIds::Time::rhs, 1);
    profiler.createValueDataSet<real>(ProfilerIds::Time::rhsElapsed, 1);
    profiler.createValueDataSet<real>(ProfilerIds::Time::loadBalancing, 1);
    profiler.createValueDataSet<real>(ProfilerIds::Time::reset, 1);
    profiler.createValueDataSet<real>(ProfilerIds::Time::removeParticles, 1);
    profiler.createValueDataSet<real>(ProfilerIds::Time::boundingBox, 1);
    profiler.createValueDataSet<real>(ProfilerIds::Time::assignParticles, 1);
    profiler.createValueDataSet<real>(ProfilerIds::Time::tree, 1);
    profiler.createValueDataSet<real>(ProfilerIds::Time::pseudoParticle, 1);
#if GRAVITY_SIM
    profiler.createValueDataSet<real>(ProfilerIds::Time::gravity, 1);
    profiler.createVectorDataSet<int>(ProfilerIds::SendLengths::gravityParticles, 1, numProcesses);
    profiler.createVectorDataSet<int>(ProfilerIds::SendLengths::gravityPseudoParticles, 1, numProcesses);
    profiler.createVectorDataSet<int>(ProfilerIds::ReceiveLengths::gravityParticles, 1, numProcesses);
    profiler.createVectorDataSet<int>(ProfilerIds::ReceiveLengths::gravityPseudoParticles, 1, numProcesses);
#endif
#if SPH_SIM
    profiler.createValueDataSet<real>(ProfilerIds::Time::sph, 1);
    profiler.createVectorDataSet<int>(ProfilerIds::SendLengths::sph, 1, numProcesses);
    profiler.createVectorDataSet<int>(ProfilerIds::ReceiveLengths::sph, 1, numProcesses);
#endif
    // Detailed timing
    profiler.createValueDataSet<real>(ProfilerIds::Time::Tree::createDomain, 1);
    profiler.createValueDataSet<real>(ProfilerIds::Time::Tree::tree, 1);
    profiler.createValueDataSet<real>(ProfilerIds::Time::Tree::buildDomain, 1);
#if GRAVITY_SIM
    profiler.createValueDataSet<real>(ProfilerIds::Time::Gravity::compTheta, 1);
    profiler.createValueDataSet<real>(ProfilerIds::Time::Gravity::symbolicForce, 1);
    profiler.createValueDataSet<real>(ProfilerIds::Time::Gravity::sending, 1);
    profiler.createValueDataSet<real>(ProfilerIds::Time::Gravity::insertReceivedPseudoParticles, 1);
    profiler.createValueDataSet<real>(ProfilerIds::Time::Gravity::insertReceivedParticles, 1);
    profiler.createValueDataSet<real>(ProfilerIds::Time::Gravity::force, 1);
    profiler.createValueDataSet<real>(ProfilerIds::Time::Gravity::repairTree, 1);
#endif
#if SPH_SIM
    profiler.createValueDataSet<real>(ProfilerIds::Time::SPH::compTheta, 1);
    profiler.createValueDataSet<real>(ProfilerIds::Time::SPH::determineSearchRadii, 1);
    profiler.createValueDataSet<real>(ProfilerIds::Time::SPH::symbolicForce, 1);
    profiler.createValueDataSet<real>(ProfilerIds::Time::SPH::sending, 1);
    profiler.createValueDataSet<real>(ProfilerIds::Time::SPH::insertReceivedParticles, 1);
    profiler.createValueDataSet<real>(ProfilerIds::Time::SPH::fixedRadiusNN, 1);
    profiler.createValueDataSet<real>(ProfilerIds::Time::SPH::density, 1);
    profiler.createValueDataSet<real>(ProfilerIds::Time::SPH::soundSpeed, 1);
    profiler.createValueDataSet<real>(ProfilerIds::Time::SPH::pressure, 1);
    profiler.createValueDataSet<real>(ProfilerIds::Time::SPH::resend, 1);
    profiler.createValueDataSet<real>(ProfilerIds::Time::SPH::internalForces, 1);
    profiler.createValueDataSet<real>(ProfilerIds::Time::SPH::repairTree, 1);
#endif
    profiler.createValueDataSet<real>(ProfilerIds::Time::integrate, 1);
    profiler.createValueDataSet<real>(ProfilerIds::Time::IO, 1);


    /// INTEGRATOR SELECTION
    // -----------------------------------------------------------------------------------------------------------------
    Miluphpc *miluphpc;
    // miluphpc = new Miluphpc(parameters, numParticles, numNodes); // not possible since abstract class
    switch (parameters.integratorSelection) {
        case IntegratorSelection::explicit_euler: {
            miluphpc = new ExplicitEuler(parameters);
        } break;
        case IntegratorSelection::predictor_corrector_euler: {
            miluphpc = new PredictorCorrectorEuler(parameters);
        } break;
        case IntegratorSelection::leapfrog: {
            miluphpc = new Leapfrog(parameters);
        } break;
        default: {
            Logger(ERROR) << "Integrator not available!";
            MPI_Finalize();
            exit(1);
        }
    }

    if (rank == 0) {
        Logger(TRACE) << "---------------STARTING---------------";
    }

    Timer timer;
    real timeElapsed;
    /// MAIN LOOP
    // -----------------------------------------------------------------------------------------------------------------
    real t = 0;
    for (int i_step=0; i_step<parameters.numOutputFiles; i_step++) {

        //profiler.setStep(i_step);

        Logger(TRACE) << "-----------------------------------------------------------------";
        Logger(TRACE, true) << "STEP: " << i_step;
        Logger(TRACE) << "-----------------------------------------------------------------";

        *miluphpc->simulationTimeHandler->h_subEndTime += (parameters.timeEnd/(real)parameters.numOutputFiles);
        Logger(DEBUG) << "subEndTime += " << (parameters.timeEnd/(real)parameters.numOutputFiles);

        miluphpc->simulationTimeHandler->copy(To::device);

        miluphpc->integrate(i_step);
        miluphpc->afterIntegrationStep();

        timer.reset();
        auto time = miluphpc->particles2file(i_step);
        timeElapsed = timer.elapsed();
        Logger(TIME) << "particles2file: " << timeElapsed << " ms";
        profiler.value2file(ProfilerIds::Time::IO, timeElapsed);


        t += parameters.timeStep;

    }

    /// END OF SIMULATION
    // -----------------------------------------------------------------------------------------------------------------
    comm.barrier();
    LOGCFG.outputRank = -1;
    if (rank == 0) {
        Logger(TRACE) << "\n\n";
        Logger(TRACE) << "---------------FINISHED---------------";
        Logger(TRACE) << "Input file: " << parameters.inputFile;
        Logger(TRACE) << "Config file: " << configFile;
        Logger(TRACE) << "Material config: " << parameters.materialConfigFile;
        if (LOGCFG.write2LogFile) {
            Logger(TRACE) << "Log file saved to: " << LOGCFG.logFileName;
        }
        if (parameters.performanceLog) {
            Logger(TRACE) << "Performance log saved to " << profilerFile.str();
        }
        if (parameters.particlesSent2H5) {
            Logger(TRACE) << "(Most recent) particles sent saved to: " << parameters.logDirectory;
        }
        Logger(TRACE) << "Generated " << parameters.numOutputFiles << " files!";
        Logger(TRACE) << "Data saved to " << parameters.directory;
        Logger(TRACE) << "---------------FINISHED---------------";
    }

    return 0;
}