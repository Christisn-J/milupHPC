#include "../../include/materials/material_handler.h"

int LibConfigReader::loadConfigFromFile(const char *configFile) {
    int numberOfElements;

    std::ifstream f(configFile);
    if (!f.good()) {
        Logger(ERROR) << "Error: config file cannot be found: " << configFile;
        MPI_Finalize();
        exit(1);
    }

    config_init(&config);

    if (!config_read_file(&config, configFile)) {
        Logger(ERROR) << "Error reading config file: " << configFile;
        const char *errorText;
        errorText = new char[500];
        errorText = config_error_text(&config);
        int errorLine = config_error_line(&config);
        Logger(ERROR) << "since: " << errorText << " on line: " << errorLine;
        delete[] errorText;
        config_destroy(&config);
        exit(1);
    }

    materials = config_lookup(&config, "materials");
    if (materials != NULL) {
        numberOfElements = config_setting_length(materials);
        int i, j;
        int maxId = 0;
        config_setting_t *material; //, *subset;

        // find max ID of materials
        for (i = 0; i < numberOfElements; ++i) {
            material = config_setting_get_elem(materials, i);
            int ID;
            if (!config_setting_lookup_int(material, "ID", &ID)) {
                Logger(ERROR) << "Error. Found material without ID in config file...";
                MPI_Finalize();
                exit(1);
            }

            Logger(DEBUG) << "Found material ID: " << ID;
            maxId = std::max(ID, maxId);
        }
        if (maxId != numberOfElements - 1) {
            Logger(ERROR) << "Material-IDs in config file have to be 0, 1, 2,...";
            MPI_Finalize();
            exit(1);
        }
    }
    return numberOfElements;
}

MaterialHandler::MaterialHandler(integer numMaterials)
        : numMaterials(numMaterials)
{
    // Allocate host memory
    h_materials = new Material[numMaterials];

    // Allocate device memory
    cuda::malloc(d_materials, numMaterials);

    // Initialize host materials with default values using the new constructors
    for (int i = 0; i < numMaterials; ++i) {
        h_materials[i] = Material((ValueSelector<ValueMode::DefaultValue, integer>*)nullptr);


        // Optional: If you want to override specific members after default construction:
        // h_materials[i].ID = ValueSelector<ValueMode::DefaultValue, integer>::value();
        // h_materials[i].interactions = ValueSelector<ValueMode::DefaultValue, integer>::value();
        // h_materials[i].sml = ValueSelector<ValueMode::DefaultValue, real>::value();
    }
}

MaterialHandler::MaterialHandler(const char *material_cfg) {
#if DEBUGGING
    Logger(DEBUG) << "Starting to load material configuration from: " << material_cfg;
#endif //DEBUGGING

    LibConfigReader libConfigReader;
    numMaterials = libConfigReader.loadConfigFromFile(material_cfg);
#if DEBUGGING
    Logger(INFO) << "Number of materials found: " << numMaterials;
#endif //DEBUGGING

    config_setting_t *material, *subset;

    h_materials = new Material[numMaterials];
#if DEBUGGING
    Logger(DEBUG) << "Allocated host memory for materials";
#endif //DEBUGGING
    cuda::malloc(d_materials, numMaterials);
#if DEBUGGING
    Logger(DEBUG) << "Allocated device memory for materials";
#endif //DEBUGGING

    for (int i = 0; i < numMaterials; ++i) {
#if DEBUGGING
        Logger(DEBUG) << "Processing material index: " << i;
#endif //DEBUGGING

        // general
        material = config_setting_get_elem(libConfigReader.materials, i);
        if (!material) {
            Logger(ERROR) << "Material setting not found at index " << i;
            continue;
        }

        idInteger id;
        if (!config_setting_lookup_int(material, "ID", &id)) {
            Logger(ERROR) << "Missing 'ID' for material index " << i;
            continue;
        }

#if DEBUGGING
        Logger(DEBUG) << "Initializing Material ID " << id << " with all parameters set to InvalidValue";
#endif //DEBUGGING
        h_materials[id] = Material((ValueSelector<ValueMode::InvalidValue, integer>*)nullptr);

#if DEBUGGING
        Logger(DEBUG) << "Material ID: " << id;
#endif //DEBUGGING

        config_setting_lookup_int(material, "ID", &id);
        h_materials[id].ID = id;
#if DEBUGGING
        Logger(DEBUG) << "Reading material data for ID: " << id;
#endif
        Logger(DEBUG) << "Reading information about material ID " << id << " out of " << numMaterials << "...";
        // material
        lookupValue(material, "interactions", &h_materials[id].interactions, id, LookupMode::Required);
        lookupValue(material, "sml", &h_materials[id].sml, id, LookupMode::Required);

#if ARTIFICIAL_VISCOSITY
        // artificial viscosity
        subset = config_setting_get_member(material, "artificial_viscosity");
        if (!subset) {
            Logger(WARN) << "Missing 'artificial_viscosity' block for material ID " << id;
        } else {
            lookupValue(subset, "alpha", &h_materials[id].artificialViscosity.alpha, id, LookupMode::Required);
            lookupValue(subset, "beta", &h_materials[id].artificialViscosity.beta, id, LookupMode::Required);
        }
#endif

#if ARTIFICIAL_STRESS
        // artificial stress
        subset = config_setting_get_member(material, "artificial_stress");
        if (!subset) {
            Logger(WARN) << "Missing 'artificial_stress' block for ID " << id;
        } else {
            lookupValue(subset, "exponent_tensor", &h_materials[id].artificialStress.exponent_tensor, id, LookupMode::Required);
            lookupValue(subset, "epsilon_stress", &h_materials[id].artificialStress.epsilon_stress, id, LookupMode::Required);
            lookupValue(subset, "mean_particle_distance", &h_materials[id].artificialStress.mean_particle_distance, id, LookupMode::Required);
        }
#endif // ARTIFICIAL_STRESS

#if PLASTICITY
        subset = config_setting_get_member(material, "plasticity");
        if (!subset) {
            Logger(WARN) << "Missing 'plasticity' block for ID " << id;
        } else {
            lookupValue(subset, "yield_stress", &h_materials[id].plasticity.yield_stress, id, LookupMode::Required);

        }
#endif

        // eos
        subset = config_setting_get_member(material, "eos");
        if (!subset) {
            Logger(ERROR) << "Missing 'eos' block for material ID " << id;
            continue;
        }

//        initializeEosValues(h_materials[id].eos);
        lookupValue(subset, "type", &h_materials[id].eos.type, id, LookupMode::Required);
#if DEBUGGING
        Logger(DEBUG) << "Parsing EOS " << h_materials[id].eos.type << " for ID: " << id;
#endif

        switch (h_materials[id].eos.type) {
            case 0: // Polytropic gas
                lookupValue(subset, "polytropic_K", &h_materials[id].eos.polytropic_K, id, LookupMode::Required);
                lookupValue(subset, "polytropic_gamma", &h_materials[id].eos.polytropic_gamma, id, LookupMode::Required);
                break;

            case 1: // Murnaghan EOS
                lookupValue(subset, "rho_0", &h_materials[id].eos.rho_0, id, LookupMode::Required);
                lookupValue(subset, "bulk_modulus", &h_materials[id].eos.bulk_modulus, id, LookupMode::Required);

                lookupValue(subset, "n", &h_materials[id].eos.n, id, LookupMode::Required);
                break;

            case 2: // Tillotson EOS
                lookupValue(subset, "rho_0", &h_materials[id].eos.rho_0, id, LookupMode::Required);
                lookupValue(subset, "bulk_modulus", &h_materials[id].eos.bulk_modulus, id, LookupMode::Required);

                lookupValue(subset, "E_0", &h_materials[id].eos.E_0, id, LookupMode::Required);
                lookupValue(subset, "till_a", &h_materials[id].eos.till_a, id, LookupMode::Required);
                lookupValue(subset, "till_b", &h_materials[id].eos.till_b, id, LookupMode::Required);
                lookupValue(subset, "till_A", &h_materials[id].eos.till_A, id, LookupMode::Required);
                lookupValue(subset, "till_B", &h_materials[id].eos.till_B, id, LookupMode::Required);
                lookupValue(subset, "till_alpha", &h_materials[id].eos.till_alpha, id, LookupMode::Required);
                lookupValue(subset, "till_beta", &h_materials[id].eos.till_beta, id, LookupMode::Required);
                lookupValue(subset, "E_iv", &h_materials[id].eos.E_iv, id, LookupMode::Required);
                lookupValue(subset, "E_cv", &h_materials[id].eos.E_cv, id, LookupMode::Required);
                lookupValue(subset, "rho_limit", &h_materials[id].eos.rho_limit, id, LookupMode::Required);
                lookupValue(subset, "cs_limit", &h_materials[id].eos.cs_limit, id, LookupMode::Required);
                break;

            case 3: // Isothermal gas
                // ggf. Parameter für die konstante Temperatur / Sound speed laden
                break;

            case 9: // Ideal gas
                lookupValue(subset, "polytropic_K", &h_materials[id].eos.polytropic_K, id, LookupMode::Required);
                lookupValue(subset, "polytropic_gamma", &h_materials[id].eos.polytropic_gamma, id, LookupMode::Required);
                break;

            case 12: // Locally isothermal gas
                // ggf. Sound speed laden
                break;

            default:
                Logger(ERROR) << "EOS type " << h_materials[id].eos.type << " not implemented.";
                break;
        }

#if SOLID
        lookupValue(subset, "shear_modulus", &h_materials[id].eos.shear_modulus, id, LookupMode::Required);
        lookupValue(subset, "bulk_modulus", &h_materials[id].eos.bulk_modulus, id, LookupMode::Required);

        if (h_materials[id].eos.bulk_modulus != InvalidValue<real>::value() && h_materials[id].eos.shear_modulus != InvalidValue<real>::value()) {
#if DIM == 3
            h_materials[id].eos.young_modulus =
                9.0 * h_materials[id].eos.bulk_modulus * h_materials[id].eos.shear_modulus /
                (3.0 * h_materials[id].eos.bulk_modulus + h_materials[id].eos.shear_modulus);
#elif DIM == 2
            h_materials[id].eos.young_modulus =
                    4.0 * h_materials[id].eos.bulk_modulus * h_materials[id].eos.shear_modulus /
                    (h_materials[id].eos.bulk_modulus + h_materials[id].eos.shear_modulus);
#else
            h_materials[id].eos.young_modulus = InvalidValue<real>::value();
#endif
        } else {
            Logger(WARN) << "Skipping calculation of Young's Modulus for Material ID " << id << " due to missing parameters.";
            lookupValue(subset, "young_modulus", &h_materials[id].eos.young_modulus, id, LookupMode::Required);
//            h_materials[id].eos.young_modulus = InvalidValue<real>::value();
        }
#else
        lookupValue(subset, "shear_modulus", &h_materials[id].eos.shear_modulus, id, LookupMode::Optional);
        lookupValue(subset, "bulk_modulus", &h_materials[id].eos.bulk_modulus, id, LookupMode::Optional);
        lookupValue(subset, "young_modulus", &h_materials[id].eos.young_modulus, id, LookupMode::Optional);

//        h_materials[id].eos.young_modulus = InvalidValue<real>::value();
#endif
    }
}

//MaterialHandler::MaterialHandler(integer numMaterials, integer ID, integer interactions, real alpha, real beta) :
//        numMaterials(numMaterials) {
//
//    h_materials = new Material[numMaterials];
//    cuda::malloc(d_materials, numMaterials);
//
//    h_materials[0].ID = ID;
//    h_materials[0].interactions = interactions;
//    //h_materials[0].artificialViscosity.alpha = 3.1;
//    h_materials[0].artificialViscosity = ArtificialViscosity(alpha, beta);
//
//}

MaterialHandler::~MaterialHandler() {

    delete[] h_materials;
    cuda::free(d_materials);

}

void MaterialHandler::copy(To::Target target, integer index) {

    if (index >= 0 && index < numMaterials) {
        cuda::copy(&h_materials[index], &d_materials[index], 1, target);
    } else {
        cuda::copy(h_materials, d_materials, numMaterials, target);
    }

}

void MaterialHandler::communicate(integer from, integer to, bool fromDevice, bool toDevice) {

    if (fromDevice) { copy(To::host); }

    boost::mpi::environment env;
    boost::mpi::communicator comm;

    //printf("numMaterials = %i    comm.rank() = %i\n", numMaterials, comm.rank());

    std::vector <boost::mpi::request> reqParticles;
    std::vector <boost::mpi::status> statParticles;


    if (comm.rank() == from) {
        reqParticles.push_back(comm.isend(to, 17, &h_materials[0], numMaterials));
    } else {
        statParticles.push_back(comm.recv(from, 17, &h_materials[0], numMaterials));
    }

    boost::mpi::wait_all(reqParticles.begin(), reqParticles.end());

    if (toDevice) { copy(To::device); }
}

void MaterialHandler::broadcast(integer root, bool fromDevice, bool toDevice) {

    if (fromDevice) { copy(To::host); }

    boost::mpi::environment env;
    boost::mpi::communicator comm;

    boost::mpi::broadcast(comm, h_materials, numMaterials, root);

    if (toDevice) { copy(To::device); }
}