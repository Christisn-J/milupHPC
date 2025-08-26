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

MaterialHandler::MaterialHandler(integer numMaterials) : numMaterials(numMaterials) {

    h_materials = new Material[numMaterials];
    cuda::malloc(d_materials, numMaterials);

    h_materials[0].ID = 0;
    h_materials[0].interactions = 0;
    //h_materials[0].artificialViscosity.alpha = 3.1;
    h_materials[0].artificialViscosity = ArtificialViscosity();
    // TODO: add for example artificial stress and other material parameters ?

}

/*MaterialHandler::MaterialHandler(const char *material_cfg) {

    LibConfigReader libConfigReader;
    numMaterials = libConfigReader.loadConfigFromFile(material_cfg);

    config_setting_t *material, *subset;

    h_materials = new Material[numMaterials];
    cuda::malloc(d_materials, numMaterials);

    double temp;

    for (int i = 0; i < numMaterials; ++i) {

        // general
        material = config_setting_get_elem(libConfigReader.materials, i);
        int id;
        config_setting_lookup_int(material, "ID", &id);
        h_materials[id].ID = id;
        Logger(DEBUG) << "Reading information about material ID " << id << " out of " << numMaterials << "...";
        config_setting_lookup_int(material, "interactions", &h_materials[id].interactions);
        config_setting_lookup_float(material, "sml", &temp);
        h_materials[id].sml = temp;

        // artificial viscosity
        subset = config_setting_get_member(material, "artificial_viscosity");
        config_setting_lookup_float(subset, "alpha", &temp);
        h_materials[id].artificialViscosity.alpha = temp;
        config_setting_lookup_float(subset, "beta", &temp);
        h_materials[id].artificialViscosity.beta = temp;

#if ARTIFICIAL_STRESS
        // artificial stress
        subset = config_setting_get_member(material, "artificial_stress");
        config_setting_lookup_float(subset, "exponent_tensor", &temp);
        h_materials[id].artificialStress.exponent_tensor = temp;
        config_setting_lookup_float(subset, "epsilon_stress", &temp);
        h_materials[id].artificialStress.epsilon_stress = temp;
        config_setting_lookup_float(subset, "mean_particle_distance", &temp);
        h_materials[id].artificialStress.mean_particle_distance = temp;
#endif

        // eos
        subset = config_setting_get_member(material, "eos");
        config_setting_lookup_int(subset, "type", &h_materials[id].eos.type);
        //config_setting_lookup_float(subset, "polytropic_K", &h_materials[id].eos.polytropic_K);
        //config_setting_lookup_float(subset, "polytropic_gamma", &h_materials[id].eos.polytropic_gamma);
        config_setting_lookup_float(subset, "polytropic_K", &temp);
        //printf("temp = %f\n", temp);
        h_materials[id].eos.polytropic_K = temp;
        config_setting_lookup_float(subset, "polytropic_gamma", &temp);
        h_materials[id].eos.polytropic_gamma = temp;

        // TODO: add switch statement (for eos type) set all needed variables accordingly, the others to -1
        *//* int config_setting_lookup_float [Function]
        (const config setting t * setting, const char * name, double * value)
         These functions look up the value of the child setting named "name" of the setting
        "setting". They store the value at "value" and return CONFIG_TRUE on success. If the
        setting was not found or if the type of the value did not match the type requested,
        they leave the data pointed to by "value" unmodified and return CONFIG_FALSE.*//*
        config_setting_lookup_float(subset, "rho_0", &temp);
        h_materials[id].eos.rho_0 = temp;
        config_setting_lookup_float(subset, "bulk_modulus", &temp);
        h_materials[id].eos.bulk_modulus = temp;
        config_setting_lookup_float(subset, "n", &temp);
        h_materials[id].eos.n = temp;
        config_setting_lookup_float(subset, "shear_modulus", &temp);
        h_materials[id].eos.shear_modulus = temp;
#if SOLID
#if DIM == 3
        // young = 9*K*mu/(3*K + mu)
        h_materials[id].eos.young_modulus = 9.0*h_materials[id].eos.bulk_modulus*h_materials[id].eos.shear_modulus / (3.0*h_materials[id].eos.bulk_modulus+h_materials[id].eos.shear_modulus);

#elif DIM == 2
        // young = 4*K_2d*mu_2d / (K_2d + mu_2D)
        h_materials[id].eos.young_modulus = 4.0*h_materials[id].eos.bulk_modulus*h_materials[id].eos.shear_modulus / (h_materials[id].eos.bulk_modulus + h_materials[id].eos.shear_modulus);
#else
        h_materials[id].eos.young_modulus = -1.0;
#endif
#else
        h_materials[id].eos.young_modulus = -1.0;
#endif

    }


}*/

MaterialHandler::MaterialHandler(const char *material_cfg) {
#if DEBUGGING
    Logger(DEBUG) << "Starting to load material configuration from: " << material_cfg;
#endif //DEBUGGING

    LibConfigReader libConfigReader;
    numMaterials = libConfigReader.loadConfigFromFile(material_cfg);
#if DEBUGGING
    Logger(DEBUG) << "Number of materials found: " << numMaterials;
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
        Logger(DEBUG) << "Material ID: " << id;
#endif //DEBUGGING

        config_setting_lookup_int(material, "ID", &id);
        h_materials[id].ID = id;
#if DEBUGGING
        Logger(DEBUG) << "Reading material data for ID: " << id;
#endif
        Logger(DEBUG) << "Reading information about material ID " << id << " out of " << numMaterials << "...";
        // material
        lookupConfigValueOrDefault(material, "interactions", &h_materials[id].interactions, id);
        lookupConfigValueOrDefault(material, "sml", &h_materials[id].sml, id);


        // artificial viscosity
        subset = config_setting_get_member(material, "artificial_viscosity");
        if (!subset) {
            Logger(WARN) << "Missing 'artificial_viscosity' block for material ID " << id;
        } else {
            lookupConfigValueOrDefault(subset, "alpha", &h_materials[id].artificialViscosity.alpha, id);
            lookupConfigValueOrDefault(subset, "beta", &h_materials[id].artificialViscosity.beta, id);
        }

#if ARTIFICIAL_STRESS
        // artificial stress
        subset = config_setting_get_member(material, "artificial_stress");
        if (!subset) {
            Logger(WARN) << "Missing 'artificial_stress' block for ID " << id;
        } else {
            lookupConfigValueOrDefault(subset, "exponent_tensor", &h_materials[id].artificialStress.exponent_tensor, id);
            lookupConfigValueOrDefault(subset, "epsilon_stress", &h_materials[id].artificialStress.epsilon_stress, id);
            lookupConfigValueOrDefault(subset, "mean_particle_distance", &h_materials[id].artificialStress.mean_particle_distance, id);
        }
#endif // ARTIFICIAL_STRESS

        // eos
        subset = config_setting_get_member(material, "eos");
        if (!subset) {
            Logger(ERROR) << "Missing 'eos' block for material ID " << id;
            continue;
        }
#if DEBUGGING
        Logger(DEBUG) << "Parsing EOS for ID: " << id;
#endif
        // TODO: add switch statement (for eos type) set all needed variables accordingly, the others to -1
        /* int config_setting_lookup_float [Function]
        (const config setting t * setting, const char * name, real * value)
         These functions look up the value of the child setting named "name" of the setting
        "setting". They store the value at "value" and return CONFIG_TRUE on success. If the
        setting was not found or if the type of the value did not match the type requested,
        they leave the data pointed to by "value" unmodified and return CONFIG_FALSE.*/

        lookupConfigValueOrDefault(subset, "type", &h_materials[id].eos.type, id);
        lookupConfigValueOrDefault(subset, "polytropic_K", &h_materials[id].eos.polytropic_K, id);
        lookupConfigValueOrDefault(subset, "polytropic_gamma", &h_materials[id].eos.polytropic_gamma, id);

        lookupConfigValueOrDefault(subset, "rho_0", &h_materials[id].eos.rho_0, id);
        lookupConfigValueOrDefault(subset, "bulk_modulus", &h_materials[id].eos.bulk_modulus, id);
        lookupConfigValueOrDefault(subset, "n", &h_materials[id].eos.n, id);
        lookupConfigValueOrDefault(subset, "shear_modulus", &h_materials[id].eos.shear_modulus, id);

#if SOLID
        if (h_materials[id].eos.bulk_modulus != -1.0 && h_materials[id].eos.shear_modulus != -1.0) {
#if DIM == 3
            h_materials[id].eos.young_modulus =
                9.0 * h_materials[id].eos.bulk_modulus * h_materials[id].eos.shear_modulus /
                (3.0 * h_materials[id].eos.bulk_modulus + h_materials[id].eos.shear_modulus);
#elif DIM == 2
            h_materials[id].eos.young_modulus =
                    4.0 * h_materials[id].eos.bulk_modulus * h_materials[id].eos.shear_modulus /
                    (h_materials[id].eos.bulk_modulus + h_materials[id].eos.shear_modulus);
#else
            h_materials[id].eos.young_modulus = -1.0;
#endif
        } else {
            Logger(WARN) << "Skipping calculation of Young's Modulus for Material ID " << id << " due to missing parameters.";
            h_materials[id].eos.young_modulus = -1.0;
        }
#else
        h_materials[id].eos.young_modulus = -1.0;
#endif


#if DEBUGGING
        Logger(DEBUG) << "Reading Tillotson parameters for material ID " << id;
#endif
        // Tillotson
        lookupConfigValueOrDefault(subset, "till_A", &h_materials[id].eos.till_A, id);
        lookupConfigValueOrDefault(subset, "till_B", &h_materials[id].eos.till_B, id);
        lookupConfigValueOrDefault(subset, "E_0", &h_materials[id].eos.E_0, id);
        lookupConfigValueOrDefault(subset, "E_iv", &h_materials[id].eos.E_iv, id);
        lookupConfigValueOrDefault(subset, "E_cv", &h_materials[id].eos.E_cv, id);
        lookupConfigValueOrDefault(subset, "till_a", &h_materials[id].eos.till_a, id);
        lookupConfigValueOrDefault(subset, "till_b", &h_materials[id].eos.till_b, id);
        lookupConfigValueOrDefault(subset, "till_alpha", &h_materials[id].eos.till_alpha, id);
        lookupConfigValueOrDefault(subset, "till_beta", &h_materials[id].eos.till_beta, id);
        lookupConfigValueOrDefault(subset, "rho_limit", &h_materials[id].eos.rho_limit, id);
        lookupConfigValueOrDefault(subset, "cs_limit", &h_materials[id].eos.cs_limit, id);

    }
}

MaterialHandler::MaterialHandler(integer numMaterials, integer ID, integer interactions, real alpha, real beta) :
        numMaterials(numMaterials) {

    h_materials = new Material[numMaterials];
    cuda::malloc(d_materials, numMaterials);

    h_materials[0].ID = ID;
    h_materials[0].interactions = interactions;
    //h_materials[0].artificialViscosity.alpha = 3.1;
    h_materials[0].artificialViscosity = ArtificialViscosity(alpha, beta);

}

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