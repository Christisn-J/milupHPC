#include "../../include/ghost/createGhostParticles.cuh"
#include "../../include/cuda_utils/cuda_launcher.cuh"
#ifndef PERIODIC_BOUNDARIES
#error "PERIODIC_BOUNDARIES is not defined!"
#endif

#if PERIODIC_BOUNDARIES

namespace SPH {

    namespace Kernel {

        __global__ void createGhostsPeriodic(Tree *tree, Particles *particles, IntegratedParticles *ghosts, integer *numGhosts){
            /// debuge
            int MODPARTICLE = 5050;

            // Note: not implemented for domain size < 2*h
            integer bodyIndex = threadIdx.x + blockIdx.x * blockDim.x;
            integer stride = blockDim.x * gridDim.x;
            integer offset = 0;

            bool foundGhostX, foundGhostY, foundGhostZ;

            // iterate over all particles
            for(offset = 0; (bodyIndex + offset) < *particles->numParticles; offset += stride){
                int iGhost = (bodyIndex + offset);

                /// debug
                if((bodyIndex + offset)%MODPARTICLE == 0){
                    printf("(?)[DEBUG]    > particle: %i \t numGhosts: %i \t iGhost: %i\n", (bodyIndex + offset), *numGhosts, iGhost);
                }
                

                if(*numGhosts >= *(particles->numParticles)*MAX_GHOSTS_PER_PARTICLE){
                    // Issue massage
                    cudaTerminate("(?)[DEBUG]    > To many ghosts at i = %i !\n %i < %i\n", (bodyIndex + offset), *(particles->numParticles)*MAX_GHOSTS_PER_PARTICLE, *numGhosts);
                }
                
                // initialis found
                foundGhostX = false;
                foundGhostY = false;
                foundGhostZ = false;

                // initialis usedGhosts
                for(int iPossible = iGhost;iPossible<*(particles->numParticles)*MAX_GHOSTS_PER_PARTICLE;iPossible+=*(particles->numParticles)) {
                    ghosts->usedGhosts[iPossible] = 1;
                }
                
/// TODO: optimize by outsourcing and parallel this initialization
                // initialis ghost map for each particle
                for(int iMap=0; iMap<MAX_GHOSTS_PER_PARTICLE; ++iMap){
                    particles->mapGhost[(bodyIndex + offset)*MAX_GHOSTS_PER_PARTICLE+iMap] = -1; //debug: -1-1*iMap;
                }
                
                // x-direction
                if (particles->x[bodyIndex + offset] <= *tree->minX + particles->sml[bodyIndex + offset]){ // && particles->x[(bodyIndex + offset)] > tree->minX) {
                    ghosts->x[iGhost] = *tree->maxX + (particles->x[bodyIndex + offset] - *tree->minX);
                    foundGhostX = true;
                } else if (*tree->maxX - particles->sml[(bodyIndex + offset)] < particles->x[(bodyIndex + offset)]){ // &&  particles->x[(bodyIndex + offset)] < tree->maxX) {
                    ghosts->x[iGhost] = *tree->minX - (*tree->maxX - particles->x[(bodyIndex + offset)]);
                    foundGhostX = true;
                } else {
                    ghosts->x[iGhost] = particles->x[(bodyIndex + offset)];
                }
                
#if DIM > 1
                // y-direction
                if (particles->y[(bodyIndex + offset)] <= *tree->minY + particles->sml[(bodyIndex + offset)]){ // && particles->y[(bodyIndex + offset)] > tree->minY) {
                    ghosts->y[iGhost] = *tree->maxY + (particles->y[(bodyIndex + offset)] - *tree->minY);
                    foundGhostY = true;
                } else if (*tree->maxY - particles->sml[(bodyIndex + offset)] < particles->y[(bodyIndex + offset)]){ // &&  particles->y[(bodyIndex + offset)] < tree->maxY) {
                    ghosts->y[iGhost] = *tree->minY - (*tree->maxY - particles->y[(bodyIndex + offset)]);
                    foundGhostY = true;
                } else {
                    ghosts->y[iGhost] = particles->y[(bodyIndex + offset)];
                }
#if DIM == 3
                // z-direction
                if (particles->z[(bodyIndex + offset)] <= *tree->minZ + particles->sml[(bodyIndex + offset)]){ // && particles->z[(bodyIndex + offset)] > tree->minZ) {
                    ghosts->z[iGhost] = *tree->maxZ + (particles->z[(bodyIndex + offset)] - *tree->minZ);
                    foundGhostZ = true;
                } else if (*tree->maxZ - particles->sml[(bodyIndex + offset)] < particles->z[(bodyIndex + offset)]){ // &&  particles->z[(bodyIndex + offset)] < tree->maxZ) {
                    ghosts->z[iGhost] = *tree->minZ - (*tree->maxZ - particles->z[(bodyIndex + offset)]);
                    foundGhostZ = true;
                } else {
                    ghosts->z[iGhost] = particles->z[(bodyIndex + offset)];
                }
#endif // 3D
#endif // 2D    

                
                /// debug
                if ((bodyIndex + offset)%MODPARTICLE == 0){
                    printf("(?)[DEBUG]    > particle: %i \t founds: %d, %d\n", (bodyIndex + offset), foundGhostX, foundGhostY);
                }
                
                
                // found boarder particle
                if (foundGhostX 
#if DIM > 1       
                    || foundGhostY 
#if DIM == 3        
                    || foundGhostZ
#endif // 3D
#endif // 2D
                ) {
                    // register ghost particle
                    particles->mapGhost[(bodyIndex + offset)*MAX_GHOSTS_PER_PARTICLE+0] = iGhost;
                    
                    // mark as required ghost 
                    ghosts->usedGhosts[iGhost] = 0;

                    /// debug
                    if ((bodyIndex + offset)%10000 == 0){
                        printf("(?)[DEBUG]    > particle: %i (%.2f | %.2f) \t ghost: %i (%.2f | %.2f) \t uesed: %d\n", (bodyIndex + offset), particles->x[bodyIndex + offset] ,particles->y[bodyIndex + offset], iGhost, ghosts->x[iGhost], ghosts->y[iGhost], ghosts->usedGhosts[iGhost]);
                    }
                    
                    // next ghost particle
                    atomicAdd(numGhosts, 1.0f);
                    iGhost +=*(particles->numParticles);
                }
                
                // found corner particle (create extra ghost particles if all are true)
                if (foundGhostX 
#if DIM > 1         
                    && foundGhostY 
#if DIM == 3        
                    && foundGhostZ
#endif // 3D
#endif // 2D
                ){
                    // fix direction
                    ghosts->y[iGhost] = particles->y[(bodyIndex + offset)];
#if DIM == 3
                    ghosts->z[iGhost] = particles->z[(bodyIndex + offset)];
#endif // 3D
                    // move in x-direction
                    if (particles->x[(bodyIndex + offset)] <= *tree->minX +  particles->sml[(bodyIndex + offset)]){ // &&  particles->x[(bodyIndex + offset)] > tree->minX) {
                        ghosts->x[iGhost] = *tree->maxX + (particles->x[(bodyIndex + offset)] - *tree->minX);
                    } else if (*tree->maxX -  particles->sml[(bodyIndex + offset)] < particles->x[(bodyIndex + offset)]){ // &&  particles->z[(bodyIndex + offset)] < tree->maxX) {
                        ghosts->x[iGhost] = *tree->minX - (*tree->maxX - particles->x[(bodyIndex + offset)]);
                    }

                    // register ghost particle
                    particles->mapGhost[(bodyIndex + offset)*MAX_GHOSTS_PER_PARTICLE+1] = iGhost;

                    // mark as required ghost 
                    ghosts->usedGhosts[iGhost] = 0;

                    /// debug
                    if ((bodyIndex + offset)%MODPARTICLE == 0){
                        printf("(?)[DEBUG]    > particle: %i (%.2f | %.2f) \t ghost: %i (%.2f | %.2f) \t uesed: %d\n", (bodyIndex + offset), particles->x[bodyIndex + offset] ,particles->y[bodyIndex + offset], iGhost, ghosts->x[iGhost], ghosts->y[iGhost], ghosts->usedGhosts[iGhost]);
                    }
                    
                    // next ghost particle
                    atomicAdd(numGhosts, 1.0f);
                    iGhost += *(particles->numParticles);

#if DIM >= 2 
                    // fix direction
                    ghosts->x[iGhost] = particles->x[(bodyIndex + offset)];
#if DIM == 3
                    ghosts->z[iGhost] = particles->z[(bodyIndex + offset)];
#endif // 3D
                    // move in y-direction
                    if (particles->y[(bodyIndex + offset)] <= *tree->minY + particles->sml[(bodyIndex + offset)]){ // && particles->y[(bodyIndex + offset)] > tree->minY) {
                        ghosts->y[iGhost] = *tree->maxY + (particles->y[(bodyIndex + offset)] - *tree->minY);
                    } else if (*tree->maxY - particles->sml[(bodyIndex + offset)] < particles->y[(bodyIndex + offset)]){ // && particles->y[(bodyIndex + offset)] < tree->maxY) {
                        ghosts->y[iGhost] = *tree->minY - (*tree->maxY - particles->y[(bodyIndex + offset)]);
                    }

                    // register ghost particle
                    particles->mapGhost[(bodyIndex + offset)*MAX_GHOSTS_PER_PARTICLE+2] = iGhost;

                    // mark as required ghost 
                    ghosts->usedGhosts[iGhost] = 0;
                    
                    /// debug
                    if ((bodyIndex + offset)%MODPARTICLE == 0){
                        printf("(?)[DEBUG]    > particle: %i (%.2f | %.2f) \t ghost: %i (%.2f | %.2f) \t uesed: %d\n", (bodyIndex + offset), particles->x[bodyIndex + offset] ,particles->y[bodyIndex + offset], iGhost, ghosts->x[iGhost], ghosts->y[iGhost], ghosts->usedGhosts[iGhost]);
                    }

                    // next ghost particle
                    atomicAdd(numGhosts, 1.0f);
                    iGhost +=*(particles->numParticles);
#endif // 2D

#if DIM == 3
                    // z-direction
                    // fix direction
                    ghosts->x[iGhost] = particles->x[(bodyIndex + offset)];
                    ghosts->y[iGhost] = particles->y[(bodyIndex + offset)];
                    // move in z-direction
                    if (particles->z[(bodyIndex + offset)] <= *tree->minZ + particles->sml[(bodyIndex + offset)]){ // && particles->z[(bodyIndex + offset)] > tree->minZ) {
                        ghosts->z[iGhost] = *tree->maxZ + (particles->z[(bodyIndex + offset)] - *tree->minZ);
                    } else if (*tree->maxY - particles->sml[(bodyIndex + offset)] < particles->z[(bodyIndex + offset)]){ // && particles->z[(bodyIndex + offset)] < tree->maxZ) {
                        ghosts->z[iGhost] = *tree->minZ - (*tree->maxZ - particles->z[(bodyIndex + offset)]);
                    }

                    // register ghost particle
                    particles->mapGhost[(bodyIndex + offset)*MAX_GHOSTS_PER_PARTICLE+3] = iGhost;

                    // mark as required ghost 
                    ghosts->usedGhosts[iGhost]= 0;

                    // next ghost particle
                    atomicAdd(numGhosts, 1.0f);
                    iGhost +=*(particles->numParticles);

/// TODO: optimization by using previous ghost particles
                    // diagonal
                    // move in x-direction
                    if (particles->x[(bodyIndex + offset)] <= *tree->minX + particles->sml[(bodyIndex + offset)]){ // && particles->x[(bodyIndex + offset)] > tree->minX) {
                        ghosts->x[iGhost] = *tree->maxX + (particles->x[(bodyIndex + offset)] - *tree->minX);
                    } else if (*tree->maxX - particles->sml[(bodyIndex + offset)] < particles->x[(bodyIndex + offset)]){ // && particles->x[(bodyIndex + offset)] < tree->maxX) {
                        ghosts->x[iGhost] = *tree->minX - (*tree->maxX - particles->x[(bodyIndex + offset)]);
                    }
                    // move in y-direction
                    if (particles->y[(bodyIndex + offset)] <= *tree->minY + particles->sml[(bodyIndex + offset)]){ // && particles->y[(bodyIndex + offset)] > tree->minY) {
                        ghosts->y[iGhost] = *tree->maxY + (particles->y[(bodyIndex + offset)] - *tree->minY);
                    } else if (*tree->maxY - particles->sml[(bodyIndex + offset)] < particles->y[(bodyIndex + offset)]){ // && particles->y[(bodyIndex + offset)] < tree->maxY) {
                        ghosts->y[iGhost] = *tree->minY - (*tree->maxY - particles->y[(bodyIndex + offset)]);
                    }
                    // move in z-direction
                    if (particles->z[(bodyIndex + offset)] <= *tree->minZ + particles->[(bodyIndex + offset)]){ // && particles->z[(bodyIndex + offset)] > tree->minZ) {
                        ghosts->z[iGhost] = *tree->maxZ + (particles->z[(bodyIndex + offset)] - *tree->minZ);
                    } else if (*tree->maxY - particles->sml[(bodyIndex + offset)] < particles->y[(bodyIndex + offset)]){ // && particles->y[(bodyIndex + offset)] < tree->maxY) {
                        ghosts->z[iGhost] = *tree->minY - (*tree->maxY - particles->y[(bodyIndex + offset)]);
                    }

                    // register ghost particle
                    particles->mapGhost[(bodyIndex + offset)*MAX_GHOSTS_PER_PARTICLE+4] = iGhost;

                    // mark as required ghost 
                    ghosts->usedGhosts[iGhost] = 0;
                    
                    // next ghost particle
                    atomicAdd(numGhosts, 1.0f);
                    iGhost +=*(particles->numParticles);  
#endif // 3D     
                }

                
                /// debug
                if ((bodyIndex + offset)%MODPARTICLE == 0){
                    printf("(?)[DEBUG]    > particle: %i \t map: (%i, %i, %i) \t uesed: (%d, %d, %d)\n", (bodyIndex + offset), particles->mapGhost[(bodyIndex + offset)*MAX_GHOSTS_PER_PARTICLE+0], particles->mapGhost[(bodyIndex + offset)*MAX_GHOSTS_PER_PARTICLE+1], particles->mapGhost[(bodyIndex + offset)*MAX_GHOSTS_PER_PARTICLE+2]), ghosts->usedGhosts[(bodyIndex + offset)+*(particles->numParticles)], ghosts->usedGhosts[(bodyIndex + offset)+*(particles->numParticles)*2],ghosts->usedGhosts[(bodyIndex + offset)+*(particles->numParticles)*3];
                }
                
                
            
            }
        }
    
        real Launch::createGhostsPeriodic(Tree *tree, Particles *particles, IntegratedParticles *ghosts, integer *numGhosts){
            ExecutionPolicy executionPolicy;
            return cuda::launch(true, executionPolicy, ::SPH::Kernel::createGhostsPeriodic, tree, particles, ghosts, numGhosts);
        }
    }
}
#endif