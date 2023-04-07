#include "../../include/ghost/createGhostParticles.cuh"

namespace SPH {

    namespace Kernel {

        __global__ void createGhostParticles(Tree *tree, Particles *particles, IntegratedParticleHandler *ghosts){
            // Note: not implemented for domain size < 2*h
            integer bodyIndex = threadIdx.x + blockIdx.x * blockDim.x;
            integer stride = blockDim.x * gridDim.x;
            
            bool foundGhostX, foundGhostY, foundGhostZ
            int iGhost = 0;

            // iterate over all particles
            for(integer offset = 0; (bodyIndex + offset) < particles->numParticles; offset += stride){

                // initialis found
                foundGhostX = false;
                foundGhostY = false;
                foundGhostZ = false;

                // initialis ghost map for each particle
/// TODO: optimize by outsourcing and parallel this initialization
                for(int iMap=0; iMap<MAX_GHOSTS_PER_PARTICLE; ++iMap){
                    particles->mapGhost[(bodyIndex + offset)*MAX_GHOSTS_PER_PARTICLE+iMap] = -1;
                }

                // x-direction
                if (particles->x[(bodyIndex + offset)] <= tree->minX + particles->sml){ // && particles->x[(bodyIndex + offset)] > tree->minX) {
                    ghosts->x[iGhost] = tree->maxX + (particles->x[(bodyIndex + offset)] - tree->minX);
                    foundGhostX = true;
                } else if (tree->maxX - particles->sml < particles->x[(bodyIndex + offset)]){ // &&  particles->x[(bodyIndex + offset)] < tree->maxX) {
                    ghosts->x[iGhost] = tree->minX - (tree->maxX - particles->x[(bodyIndex + offset)]);
                    foundGhostX = true;
                } else {
                    ghosts->x[iGhost] = particles->x[(bodyIndex + offset)];
                }

#if DIM > 1
                // y-direction
                if (particles->y[(bodyIndex + offset)] <= tree->minY + particles->sml){ // && particles->y[(bodyIndex + offset)] > tree->minY) {
                    ghosts->y[iGhost] = tree->maxY + (particles->y[(bodyIndex + offset)] - tree->minY);
                    foundGhostY = true;
                } else if (tree->maxY - particles->sml < particles->y[(bodyIndex + offset)]){ // &&  particles->y[(bodyIndex + offset)] < tree->maxY) {
                    ghosts->y[iGhost] = tree->minY - (tree->maxY - particles->y[(bodyIndex + offset)]);
                    foundGhostY = true;
                } else {
                    ghosts->y[iGhost] = particles->Y[(bodyIndex + offset)];
                }
#if DIM == 3
                // z-direction
                if (particles->z[(bodyIndex + offset)] <= tree->minZ + particles->sml){ // && particles->z[(bodyIndex + offset)] > tree->minZ) {
                    ghosts->z[iGhost] = tree->maxZ + (particles->z[(bodyIndex + offset)] - tree->minZ);
                    foundGhostZ = true;
                } else if (tree->maxZ - particles->sml < particles->z[(bodyIndex + offset)]){ // &&  particles->z[(bodyIndex + offset)] < tree->maxZ) {
                    ghosts->z[iGhost] = tree->minZ - (tree->maxZ - particles->z[(bodyIndex + offset)]);
                    foundGhostZ = true;
                } else {
                    ghosts->z[iGhost] = particles->z[(bodyIndex + offset)];
                }
#endif // 3D
#endif // 2D

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
                    particles->mapGhost[i*MAX_GHOSTS_PER_PARTICLE+0] = iGhost;

                    // next ghost particle
                    iGhost = atomicAdd(iGhost, 1);
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
                    if (particles->x[(bodyIndex + offset)] <= tree->minX +  particles->sml){ // &&  particles->x[(bodyIndex + offset)] > tree->minX) {
                        ghosts->x[iGhost] = tree->maxX + (particles->z[(bodyIndex + offset)] - tree->minX);
                    } else if (tree->maxX -  particles->sml < particles->z[(bodyIndex + offset)]){ // &&  particles->z[(bodyIndex + offset)] < tree->maxX) {
                        ghosts->x[iGhost] = tree->minX - (tree->maxX - particles->z[(bodyIndex + offset)]);
                    }

                    // register ghost particle
                    particles->mapGhost[i*MAX_GHOSTS_PER_PARTICLE+1] = iGhost;

                    // next ghost particle
                    iGhost = atomicAdd(iGhost, 1);

#if DIM >= 2 
                    // fix direction
                    ghosts->x[iGhost] = particles->x[(bodyIndex + offset)];
#if DIM == 3
                    ghosts->z[iGhost] = particles->z[(bodyIndex + offset)];
#endif // 3D
                    // move in y-direction
                    if (particles->y[(bodyIndex + offset)] <= tree->minY + particles->sml){ // && particles->y[(bodyIndex + offset)] > tree->minY) {
                        ghosts->y[iGhost] = tree->maxY + (particles->y[(bodyIndex + offset)] - tree->minY);
                    } else if (tree->maxY - particles->sml < particles->y[(bodyIndex + offset)]){ // && particles->y[(bodyIndex + offset)] < tree->maxY) {
                        ghosts->y[iGhost] = tree->minY - (tree->maxY - particles->y[(bodyIndex + offset)]);
                    }

                    // register ghost particle
                    particles->mapGhost[i*MAX_GHOSTS_PER_PARTICLE+2] = iGhost;

                    // next ghost particle
                    iGhost = atomicAdd(iGhost, 1);
#endif // 2D

#if DIM == 3
                    // z-direction
                    // fix direction
                    ghosts->x[iGhost] = particles->x[(bodyIndex + offset)];
                    ghosts->y[iGhost] = particles->y[(bodyIndex + offset)];
                    // move in z-direction
                    if (particles->z[(bodyIndex + offset)] <= tree->minZ + particles->sml){ // && particles->z[(bodyIndex + offset)] > tree->minZ) {
                        ghosts->z[iGhost] = tree->maxZ + (particles->z[(bodyIndex + offset)] - tree->minZ);
                    } else if (tree->maxY - particles->sml < particles->z[(bodyIndex + offset)]){ // && particles->z[(bodyIndex + offset)] < tree->maxZ) {
                        ghosts->z[iGhost] = tree->minZ - (tree->maxZ - particles->z[(bodyIndex + offset)]);
                    }

                    // register ghost particle
                    particles->mapGhost[i*MAX_GHOSTS_PER_PARTICLE+3] = iGhost;


                    // next ghost particle
                    iGhost = atomicAdd(iGhost, 1);

/// TODO: optimization by using previous ghost particles
                    // diagonal
                    // move in x-direction
                    if (particles->x[(bodyIndex + offset)] <= tree->minX + particles->sml){ // && particles->x[(bodyIndex + offset)] > tree->minX) {
                        ghosts->x[iGhost] = tree->maxX + (particles->x[(bodyIndex + offset)] - tree->minX);
                    } else if (tree->maxX - particles->sml < particles->x[(bodyIndex + offset)]){ // && particles->x[(bodyIndex + offset)] < tree->maxX) {
                        ghosts->x[iGhost] = tree->minX - (tree->maxX - particles->x[(bodyIndex + offset)]);
                    }
                    // move in y-direction
                    if (particles->y[(bodyIndex + offset)] <= tree->minY + particles->sml){ // && particles->y[(bodyIndex + offset)] > tree->minY) {
                        ghosts->y[iGhost] = tree->maxY + (particles->y[(bodyIndex + offset)] - tree->minY);
                    } else if (tree->maxY - particles->sml < particles->y[(bodyIndex + offset)]){ // && particles->y[(bodyIndex + offset)] < tree->maxY) {
                        ghosts->y[iGhost] = tree->minY - (tree->maxY - particles->y[(bodyIndex + offset)]);
                    }
                    // move in z-direction
                    if (particles->z[(bodyIndex + offset)] <= tree->minZ + particles->sml){ // && particles->z[(bodyIndex + offset)] > tree->minZ) {
                        ghosts->z[iGhost] = tree->maxZ + (particles->z[(bodyIndex + offset)] - tree->minZ);
                    } else if (tree->maxY - particles->sml < particles->y[(bodyIndex + offset)]){ // && particles->y[(bodyIndex + offset)] < tree->maxY) {
                        ghosts->z[iGhost] = tree->minY - (tree->maxY - particles->y[(bodyIndex + offset)]);
                    }

                    // register ghost particle
                    particles->mapGhost[i*MAX_GHOSTS_PER_PARTICLE+4] = iGhost;


                    // next ghost particle
                    iGhost = atomicAdd(iGhost, 1);    
#endif // 3D
                }
            }
        // remaind the total number ghost particles
        ghosts->numParticles = iGhost;
        }

        Particles Launch::createGhostParticles(Tree *tree, Particles *particles, IntegratedParticleHandler *ghosts){
            ExecutionPolicy executionPolicy;
            return cuda::launch(true, executionPolicy, ::SPH::Kernel::createGhostParticles, tree, particles, ghosts);
        }
    }
}