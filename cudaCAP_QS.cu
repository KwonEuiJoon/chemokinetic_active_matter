#define R0 (1.0)
#define R02 (1.0)

#define MaxThreads (512)

#include "subCAP_QS.c" 
#include <sys/stat.h>
#include <string.h>

void save_current_state(struct act_ptl *Ptls, const int ptlsNum, char *filename);
void save_chem(float *chem, const int cllsNum, char *filename);

int main(int argc, char *argv[])
{
        // device setting
        const int    device_num = atoi(argv[1]);
        if(device_num<0 || device_num>3) error_output("invalid device number") ;
        cudaSetDevice(device_num);

        // folder setting
        char folder_name[100];
        sprintf(folder_name, "%s", argv[2]);
        mkdir(folder_name, S_IRWXU);

        // file name setting
        char file_name[100];
        sprintf(file_name, "%s", argv[2]);

        // Lattice and cell setting
        const int    Lx_size  = atoi(argv[3]);
        const int    Ly_size  = atoi(argv[4]);
        const int    tmax   = atoi(argv[5]);
        const int    csize  = 1;
    
        // Dynamics setting
        const float  dt     = 0.001;
        const float  rho    = atof(argv[6]);
        const float  n0     = atof(argv[7]);
        const float  zeta   = atof(argv[8]);
        const float  lm     = atof(argv[9]);
        const int    mode   = atoi(argv[10]);
        float inj;
        if (mode == 0) { inj = 0.0; }
        else if (mode == 1) {inj = lm * rho * n0; }
        else if (mode == 2) {inj = lm * rho * (n0 - 1.0* zeta * rho) * n0; }
        else {error_output("invalid mode") ;}
        printf("%f\n", inj);

        // time measurement
        clock_t start, end;
        float res;

        start = clock();


        // total number of particles
        const  int   ptlsNum = (int)(Lx_size*Ly_size*rho) ;
    
        // Set noise strength
        float D0 = 1.0; // no thermal diffusion
        float Dr = 3.0;
        const float Dc = atof(argv[11]);

        // hyper-parameter save
        char hp_filename[200];
        char len_filename[200];
        sprintf(hp_filename, "%s/hyper_parameters.txt", folder_name);    
        FILE *fp = fopen(hp_filename, "w");
                if (fp == NULL) {printf("Error opening the file %s", hp_filename); return 0;}
        fprintf(fp, "Lx Ly tmax csize dt rho n0 zeta lm D0 Dr Dc\n");
        fprintf(fp, "%d %d %d %d %f %f %f %f %f %f %f %f", Lx_size, Ly_size, tmax, csize, dt, rho, n0, zeta, lm, D0, Dr, Dc);

        fclose(fp);
        
        // load saved configurations
        char ptls_data[2000000];
        char chem_data[500000];
        
        if (argc == 14) 
            {   char saved_folder_name[200];
                char saved_ptls_name[200];
                char saved_chem_name[200];
                
                sprintf(saved_ptls_name, "%s/state_t_%d.txt", argv[12], atoi(argv[13]));
                sprintf(saved_chem_name, "%s/chem_t_%d.txt", argv[12], atoi(argv[13]));
                
                FILE *sv_ptls = fopen(saved_ptls_name, "r");
                if (sv_ptls == NULL) {printf("Error opening the savedfile %s", saved_ptls_name); return 0;}
                
                fgets(ptls_data, sizeof(ptls_data), sv_ptls);
                fclose(sv_ptls);
             
                FILE *sv_chem = fopen(saved_chem_name, "r");
                if (sv_chem == NULL) {printf("Error opening the savedfile %s", saved_chem_name); return 0;}
                
                fgets(chem_data, sizeof(chem_data), sv_chem);
                fclose(sv_chem);
                
                
             }
        
        

        // total number of cells
        // cell size 2*2
        const  int   cllsNum    = Lx_size*Ly_size ;

        // grid dimension
        // for particle
        const int nThreads = (MaxThreads<ptlsNum)? MaxThreads : ptlsNum;
        const int nBlocks  = (ptlsNum+nThreads-1)/nThreads; 
        // for chemical field
        const int nThreads2 = (MaxThreads<cllsNum)? MaxThreads : cllsNum;
        const int nBlocks2   = (cllsNum+nThreads2-1)/nThreads2;
        
        // active particle
        struct act_ptl *devPtls;
        cudaMalloc(&devPtls, sizeof(struct act_ptl)*ptlsNum) ;
        // chemical field
        float *devChem;
        cudaMalloc(&devChem, sizeof(float)*cllsNum);
    
        // auxiliary memory for linked lists
        // linked list is managed with the THRUST library
        // corresponding device memory
        int *devCell, *devHead, *devTail ;
        cudaMalloc(&devCell, sizeof(int)*ptlsNum);
        cudaMalloc(&devHead, sizeof(int)*cllsNum);
        cudaMalloc(&devTail, sizeof(int)*cllsNum);
    
        // get current state
        size_t memSize = sizeof(struct act_ptl) * ptlsNum;
        struct act_ptl *hostPtls;
        hostPtls = (struct act_ptl *)malloc(memSize);
        float *hostChem;
        hostChem = (float *)malloc(sizeof(float) * cllsNum);
    
        // set the PRNG seed with the device random number
        std::random_device rd;
        unsigned int seed = rd();
        // seed = 1234;
        // initialize the PRNGs
        curandState *devStates ;
        cudaMalloc((void **)&devStates, ptlsNum*sizeof(curandState)) ;
        initialize_prng<<<nBlocks, nThreads>>>(ptlsNum, seed, devStates) ;
    
    
        // random initial configuration
        if (argc == 12) {
            init_random_config(hostPtls, Lx_size, Ly_size, csize, ptlsNum);
            cudaMemcpy(devPtls, hostPtls, memSize, cudaMemcpyHostToDevice);
            init_chem<<<nBlocks2, nThreads2>>>(devChem, cllsNum, n0);
        }
        
        if (argc == 14) {
            init_from_config(hostPtls, hostChem, ptls_data, chem_data, ptlsNum, cllsNum);
            cudaMemcpy(devPtls, hostPtls, memSize, cudaMemcpyHostToDevice);
            cudaMemcpy(devChem, hostChem, sizeof(float) * cllsNum, cudaMemcpyHostToDevice);
        }
            
        // thermalization
//         for(int t = 0; t<=1000000; t++) {
//             // linked list
//             linked_list(devPtls, Lx_size, Ly_size, csize, ptlsNum, cllsNum, devCell, devHead, devTail, nBlocks, nThreads);
//             // particle move
//             move<<<nBlocks, nThreads>>>(devPtls, devChem, devStates, devHead, devTail, Lx_size, Ly_size, csize, ptlsNum, dt, zeta, D0, Dr);
//             if (mode == 1) {chem_update_rhon<<<nBlocks2, nThreads2>>>(devChem, devHead, devTail, Lx_size, Ly_size, csize, cllsNum, dt, Dc, n0, lm, rho, inj); }
//             if (mode == 2) {chem_update_rhovn<<<nBlocks2, nThreads2>>>(devPtls, devChem, devHead, devTail, Lx_size, Ly_size, csize, cllsNum, dt, Dc, n0, lm, rho, inj); }
            
//         }

        
        // measurement
        for(int t = 0; t<=tmax; t++) {
            // linked list
            linked_list(devPtls, Lx_size, Ly_size, csize, ptlsNum, cllsNum, devCell, devHead, devTail, nBlocks, nThreads);
            // particle move
            move<<<nBlocks, nThreads>>>(devPtls, devChem, devStates, devHead, devTail, Lx_size, Ly_size, csize, ptlsNum, dt, zeta, D0, Dr);
            if (mode == 1) {chem_update_rhon<<<nBlocks2, nThreads2>>>(devChem, devHead, devTail, Lx_size, Ly_size, csize, cllsNum, dt, Dc, n0, lm, rho, inj); }
            if (mode == 2) {chem_update_rhovn<<<nBlocks2, nThreads2>>>(devPtls, devChem, devHead, devTail, Lx_size, Ly_size, csize, cllsNum, dt, Dc, n0, lm, rho, inj); }
            
            if (t % 2500 == 0){
                printf("t = %d\n", t);
                // Get current state
                cudaMemcpy(hostPtls, devPtls, memSize, cudaMemcpyDeviceToHost);
                cudaMemcpy(hostChem, devChem, sizeof(float) * cllsNum, cudaMemcpyDeviceToHost);
        
                char state_filename[200];
                char chem_filename[200];
                sprintf(state_filename, "%s/state_t_%d.txt", folder_name, t);
                sprintf(chem_filename, "%s/chem_t_%d.txt", folder_name, t);
                save_current_state(hostPtls, ptlsNum, state_filename);
                save_chem(hostChem, cllsNum, chem_filename);
                
            }
        }
    
    cudaFree(devPtls); cudaFree(devChem); cudaFree(devStates); cudaFree(devCell); cudaFree(devHead); cudaFree(devTail); cudaFree(devChem);
    free(hostPtls); free(hostChem);

    
}


void save_current_state(struct act_ptl *Ptls, const int ptlsNum, char *filename){
        FILE *fp = fopen(filename, "w");
        if (fp == NULL) {printf("Error opening the file %s", filename); return;}
        
        for (int i = 0; i < ptlsNum; i++) {
                fprintf(fp, "%f %f %f %f %f ", Ptls[i].px, Ptls[i].py, Ptls[i].vx, Ptls[i].vy, Ptls[i].angle);
        }
        fprintf(fp, "\n");
        fclose(fp);
        
}

void save_chem(float *chem, const int cllsNum, char *filename){
        FILE *fp = fopen(filename, "w");
        if (fp == NULL) {printf("Error opening the file %s", filename); return;}
        
        for (int i = 0; i < cllsNum; i++) {
                fprintf(fp, "%f ", chem[i]);
        }
        fprintf(fp, "\n");
        fclose(fp);
        
}
