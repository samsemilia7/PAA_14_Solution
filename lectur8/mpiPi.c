#include <stdio.h>
#include <math.h>
#include <mpi.h>
/**/
#define ROUNDS 5
/* a known highly accurate approximation of Pi*/
double PI25DT = 3.141592653589793238462643;
/**/
int main( int argc, char *argv[] )
{    
    /* Initialization */
    /*  Notice: the number of processes is set 
        by the argument '-np' of the mpirun command line (argv)
        in this practical, this number is set to 4*/
    MPI_Init(&argc,&argv);
    /**/
    int n; /* number of intervals */
    int myid; /* the identifier of the process */
    int numprocs; /* number of processes */
    int i,j; 
    double mypi, pi, h, sum, x; 
    /* get value of myid and numprocs */
    MPI_Comm_rank(MPI_COMM_WORLD,&myid);
    MPI_Comm_size(MPI_COMM_WORLD,&numprocs);
    /* display the number of processes, just to check */
    if(myid == 0)
    {
        printf("number of processes: %d\n",numprocs);   
    }
    
    /*  we perform multiple rounds, 
        with an increasing n to evaluate the computation accuracy */
    for(j=0;j<ROUNDS;++j)
    {
        /* emulate user input */
        if(myid == 0)
        {
            n = pow(10, j+1); /* n = 10, 100, ...*/
        }
        /***************************************/
        /* YOUR TASK (with 3 TODOs) STARTS HERE*/
        /***************************************/
        
        /* Your TODO-1 starts here 
           Broadcast 'n' from process 0 */
        MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
        /* Your TODO-1 ends here*/
        
        

        /* Your TODO-2 starts here 
           Compute local 'mypi' */
        h = 1.0 / (double) n; 
        sum = 0.0; 
        for (i = myid + 1; i <= n; i += numprocs) 
        { 
            x = h * ((double)i - 0.5); 
            sum += 4.0 / (1.0 + x*x); 
        } 
        mypi = h * sum;     

        /* Your TODO-2 ends here*/
        


        /* Your TODO-3 starts here
         Reduce local 'mypi' to value 'pi' of process 0 */
        MPI_Reduce(&mypi, &pi, 1, MPI_DOUBLE, MPI_SUM, 0,MPI_COMM_WORLD);
        /* Your TODO-3 ends here*/
        
        /***********************/
        /* YOUR TASK ENDS HERE */
        /***********************/
        
        /* Print the computation Pi 
            and its difference from the high accuracy Pi */
        if (myid == 0)
        {
            printf("n = %d, pi is approx: %.16f, Error is %.16f\n",n,pi, fabs(pi-PI25DT));
        } 
    } /*End of the loop through different values of n */
    MPI_Finalize(); 
    return 0; 
}
