#include <math.h>
#include <unistd.h>
#include <random>
#include <iostream>
#include <cuda.h>
#include <curand_kernel.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/device_malloc.h>
#include <thrust/sort.h>
#include <thrust/binary_search.h>
#include <thrust/adjacent_difference.h>
#include <cuda_runtime.h>

#define two_ppi (6.28318530717958648)
#define ppi (3.14159265358979324)

struct act_ptl {
    float px, py, vx, vy, angle, size;
} ;

void error_output(const char *desc)
{
    printf("%s\n", desc) ; exit(-1) ;
}

// initializing RNG for all threads with the same seed
// each state setup will be the state after 2^{67}*tid calls 
__global__ void initialize_prng(const int ptlsNum, 
        unsigned int seed, curandState *state)
{
    const int tid = threadIdx.x + blockDim.x * blockIdx.x ;
    if(tid<ptlsNum)
        curand_init(seed, tid, 0, &state[tid]) ;
}

void init_random_config(struct act_ptl *ptls, const int lx, const int ly, const int csize, const int ptlsNum){
	int x, y;
	
	int pos[lx*ly];
	for (int i=0;i<lx*ly;i++) {pos[i] = 0;}
    
    
	// Initialize active particles
	for (int i=0;i<ptlsNum;i++) {
        do {
            x = (int) floor(((double) rand() / RAND_MAX)*(lx-1)+1);
            y = (int) floor(((double) rand() / RAND_MAX)*(ly-1)+1);
        } while (pos[x+y*lx]);
        ptls[i].px = x*csize;
        ptls[i].py = y*csize;
        pos[x+y*lx] = 1;

        ptls[i].angle = two_ppi*((double) rand() / RAND_MAX);
        ptls[i].vx = 0.;
        ptls[i].vy = 0.;
        ptls[i].size = 1.0;
	}
    
}

__global__ void init_chem(float *chem, const int cllsNum, const float n0)
{
    const int tid = threadIdx.x + blockDim.x * blockIdx.x ;
    if(tid < cllsNum) {
        chem[tid] = n0;
    }
}


void init_from_config(struct act_ptl *ptls, float *chem, char *ptls_data, char *chem_data, const int ptlsNum, const int cllsNum)
{
    char delim[] = " ";
    char *ptr = strtok(ptls_data, delim);
    for (int i=0; i<ptlsNum; i++) {
        ptls[i].px = atof(ptr);
        ptr = strtok(NULL, delim);
        ptls[i].py = atof(ptr);
        ptr = strtok(NULL, delim);
        ptls[i].vx = atof(ptr);
        ptr = strtok(NULL, delim);
        ptls[i].vy = atof(ptr);
        ptr = strtok(NULL, delim);
        ptls[i].angle = atof(ptr);
        ptr = strtok(NULL, delim);
    }
    
    char *ptr2 = strtok(chem_data, delim);
    for (int i=0; i<cllsNum; i++){
        chem[i] = atof(ptr2);
        ptr2 = strtok(NULL, delim);
    }
    
}

        

__global__ void move(struct act_ptl *ptls, float *chem, curandState *state, int *cellHead, int *cellTail, const int lx, const int ly, const int csize, const int ptlsNum, const float dt, const float u0, const float D0, const float Dr)
{
    const int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if(tid < ptlsNum) {
        
        // calculating force
        float x = ptls[tid].px, y = ptls[tid].py, theta = ptls[tid].angle;
        float Fx = 0.0, Fy = 0.0;
        float F, dx, dy, dr, v0;
        float Rs = pow(2.0, -1.0/6.0) * R0;
        
        for(int a=(int) (x/csize)-1; a<=(int) (x/csize) +1; a++) {
            for(int b=(int) (y/csize)-1; b<=(int) (y/csize) +1; b++) {
                // zz : index for neighboring cells
                int zz = (a+lx)%lx + ((b+ly)%ly)*lx;
                
                for(int k=cellHead[zz]; k<=cellTail[zz]; k++) {
                    // loop over particles in the cell zz
                    dx = fmodf(ptls[k].px - x + 3 * (float)(lx/2.), (float)lx) - (float)(lx/2.); // the first variable of fmodf should be positive
                    dy = fmodf(ptls[k].py - y + 3 * (float)(ly/2.), (float)ly) - (float)(ly/2.);
                    
                    dr = sqrt(dx*dx + dy*dy);
                    if(dr*dr < R02 && dr != 0) {
                        F = 4 * u0 * (-12.0 * pow(Rs, 12.0) * pow(dr, -13.0) + 6.0 * pow(Rs, 6.0) * pow(dr, -7.0));
                        // F = u0 * (dr - R0) / R0;
                        Fx += F * dx/dr;
                        Fy += F * dy/dr;
                        
                    }
                }
            }
        }
        
        // activity calculation
        int pos = (int) (x/csize) + ((int) (y/csize))*lx;
        v0 = chem[pos];
         
        ptls[tid].vx = v0 * cosf(theta) + Fx + sqrt(2.0 *D0 / dt) *(2.*curand_uniform(&state[tid])-1.0);
        ptls[tid].vy = v0 * sinf(theta) + Fy + sqrt(2.0 *D0 / dt) *(2.*curand_uniform(&state[tid])-1.0);
        
        //updating X
        ptls[tid].px += dt * ptls[tid].vx;
        if(ptls[tid].px < 0.0)    ptls[tid].px += lx ;
        if(ptls[tid].px >= lx) ptls[tid].px -= lx ;

        //updating Y
        ptls[tid].py += dt * ptls[tid].vy;
        if(ptls[tid].py < 0.0)    ptls[tid].py += ly ;
        if(ptls[tid].py >= ly) ptls[tid].py -= ly ;
        
        //updating angle
        ptls[tid].angle += dt * (sqrt(2.0*Dr / dt) * (2.*curand_uniform(&state[tid])-1.0));
        if(ptls[tid].angle < -1. * ppi)   ptls[tid].angle += two_ppi;
        if(ptls[tid].angle > ppi)         ptls[tid].angle -= two_ppi;
        
    }
    
}

__global__ void chem_update_BMR(float *chem, int *cellHead, int *cellTail, const int lx, const int ly, const int csize, const int cllsNum, const float dt, const float Dc, const float n0, const float lm, const float rho, float inj)
{
    const int tid = threadIdx.x + blockDim.x * blockIdx.x ;
    if(tid<cllsNum) {
        int rho_loc; //local density at tid
        if (cellTail[tid] == 0 & cellHead[tid] == -1) { rho_loc = 0; }
        else { rho_loc = cellTail[tid] - cellHead[tid] + 1; }
        
        int x = tid%lx;
        int y = tid/lx;
        
        chem[x + y*lx] += dt * Dc * (-0.5 * ( chem[(x-1+lx)%lx + ((y-1+ly)%ly) * lx] + chem[(x+1+lx)%lx + ((y-1+ly)%ly) * lx] + chem[(x-1+lx)%lx + ((y+1+ly)%ly) * lx] + chem[(x+1+lx)%lx + ((y+1+ly)%ly) * lx]) + 2.0 * ( chem[x + ((y-1+ly)%ly) * lx] + chem[x + ((y+1+ly)%ly) * lx] + chem[(x-1+lx)%lx + y * lx] + chem[(x+1+lx)%lx + y * lx]) - 6.0 * chem[x + y *lx]) / R02  + dt * (inj - lm* (float) rho_loc * chem[x + y*lx]);
        
    
        
    }
}

__global__ void chem_update_AMR(struct act_ptl *ptls, float *chem, int *cellHead, int *cellTail, const int lx, const int ly, const int csize, const int cllsNum, const float dt, const float Dc, const float n0, const float lm, const float rho, float inj)
{
    const int tid = threadIdx.x + blockDim.x * blockIdx.x ;
    if(tid<cllsNum) {
        float rv_loc = 0.0; //local density at tid
        if (cellTail[tid] == 0 & cellHead[tid] == -1) { rv_loc = 0.0; }
        else {
            for(int k=cellHead[tid]; k<=cellTail[tid]; k++){
                rv_loc += ptls[k].vx * cosf(ptls[k].angle) + ptls[k].vy * sinf(ptls[k].angle);
            }
        }
        
        int x = tid%lx;
        int y = tid/lx;
        
        chem[x + y*lx] += dt * Dc * (-0.5 * ( chem[(x-1+lx)%lx + ((y-1+ly)%ly) * lx] + chem[(x+1+lx)%lx + ((y-1+ly)%ly) * lx] + chem[(x-1+lx)%lx + ((y+1+ly)%ly) * lx] + chem[(x+1+lx)%lx + ((y+1+ly)%ly) * lx]) + 2.0 * ( chem[x + ((y-1+ly)%ly) * lx] + chem[x + ((y+1+ly)%ly) * lx] + chem[(x-1+lx)%lx + y * lx] + chem[(x+1+lx)%lx + y * lx]) - 6.0 * chem[x + y *lx]) / R02  + dt * (inj - lm* rv_loc * chem[x + y*lx]);
        
        // chem[x + y*lx] += dt * (- 0.01 * (float) rho_loc * chem[x + y*lx]);
        
    }
}

// make a table "cell[i]" for the cell index for a particle i
__global__ void find_address(struct act_ptl *ptls, 
        const int lx, const int ly, const int csize, const int ptlsNum, int *cell)
{
    const int tid = threadIdx.x + blockDim.x * blockIdx.x ;
    if(tid<ptlsNum) {
         cell[tid] = ((int) (ptls[tid].px/csize))%lx 
                    + lx*(((int) (ptls[tid].py/csize))%ly) ;
    }
}


// make tables "cellHead[c]" and "cellTail[c]" for the index 
// of the first and the last praticle in a cell c
// empty cells are not updated
__global__ void cell_head_tail(int ptlsNum, int *cell, 
        int *cellHead, int *cellTail)
{
    const int tid = threadIdx.x + blockDim.x * blockIdx.x ;
    if(tid<ptlsNum) {
        if(tid==0) cellHead[cell[tid]] = tid ;
        else {
            if(cell[tid]!=cell[tid-1]) cellHead[cell[tid]] = tid ;
        }
        if(tid==ptlsNum-1) cellTail[cell[tid]] = tid ;
        else {
            if(cell[tid]!=cell[tid+1]) cellTail[cell[tid]] = tid ;
        }
    }
}


void linked_list(struct act_ptl *ptls, const int lx, const int ly, const int csize,
        const int ptlsNum, const int cllsNum, int *cell,  
        int *head, int *tail, int nBlocks, int nThreads)
{
    // cell[ptl] = cell index of a particle
    find_address<<<nBlocks, nThreads>>>(ptls, lx, ly, csize, ptlsNum, cell);
    // sort particles w.r.t the cell index
    thrust::sort_by_key(thrust::device_ptr<int>(cell),
                thrust::device_ptr<int>(cell)+ptlsNum,
                thrust::device_ptr<struct act_ptl>(ptls));
    thrust::fill(thrust::device_ptr<int>(head),
            thrust::device_ptr<int>(head)+cllsNum, 0);
    thrust::fill(thrust::device_ptr<int>(tail),
            thrust::device_ptr<int>(tail)+cllsNum, -1);
    // find the first (head) and the last (tail)  particle indices in each cell
    // head = -1 and tail = 0 for the empty cell
    cell_head_tail<<<nBlocks, nThreads>>>(ptlsNum, cell, head, tail);
}


        