#include <stdlib.h>
#include <float.h>
#include <math.h>

#include "mex.h"

#ifdef _OPENMP    
#include <omp.h>
#endif

void minmedmax(const long R, const long C, double *X, double *Y, double *I) {
    long g, h, i, j, lb, nlb, ub, nub, t[2], M;
    double pv, lpv, upv, *Xj, *Yj, Ygj, Yhj, *Ij, Igj, Ihj, cr, mn, mx;

    /* index of median */
    M = R/2; /* equivalent to floor(R/2); use floor instead of ceil due to C indexing >< Matlab indexing */
    
    /* initialise each column of Y and I */
    #pragma omp parallel for private(Yj,Ij,g,h,i,cr,mn,mx)
    for (j = 0; j<C; j++) {   
        /* offset */
        Xj = &X[R*j];
        Yj = &Y[R*j];
        Ij = &I[R*j];
        
        /* initialise min and max */
        g = 0;
        mn = Xj[g];
        h = R - 1;
        mx = Xj[h];
        
        /* for each row */
        for (i = 0; i<R; i++) {
            /* current value */
            cr = Xj[i];
            
            /* initialisation */
            Yj[i] = cr;
            Ij[i] = (double)(i+1);
            
            /* update min and max if necessary */
            if (cr<mn) { g = i; mn = cr; } else if (cr>mx) { h = i; mx = cr; }
        }
        
        /* swap min and max; min first, max second */
        cr = Yj[  0]; Yj[  0] = Yj[g]; Yj[g] = cr;
        i  = Ij[  0]; Ij[  0] = Ij[g]; Ij[g] = i;
        if (h==0) h = g; /* track max if already swapped with min */
        cr = Yj[R-1]; Yj[R-1] = Yj[h]; Yj[h] = cr;
        i  = Ij[R-1]; Ij[R-1] = Ij[h]; Ij[h] = i;
    }
    
    /* exit if nothing to do */
    if (R<=3) return;
    
    /* for each column (difficult to parallelise: many variables, vector t => matrix) */
    for (j = 0; j<C; j++) {   
        /* offset */
        Yj = &Y[R*j];
        Ij = &I[R*j];
        
        /* initialise bounds of first chunk (exclude min and max, already found) */
        t[0] = 1;
        t[1] = R - 2;
        
        /* while chunks are left */
        for ( ; ; ) {
            /* lower and upper bounds */
            lb = t[0];
            ub = t[1];
            
            /* three possible random pivot values => take the intermediate one */
            lpv = Yj[lb]; /* first */
            upv = Yj[ub]; /* last */
            if (upv<lpv) { pv = lpv; lpv = upv; upv = pv; } /* sort */
            pv = Yj[(lb+ub)/2]; /* middle */
            if (pv<lpv) pv = lpv; else if (pv>upv) pv = upv; /* pivot is the intermediate */

            /* swap around pivot */
            for (g = lb, h = ub; g<h; ) {
                /* compare left value to pivot */
                Ygj = Yj[g];
                Igj = Ij[g];
                if (Ygj<pv) {
                    /* OK left : keep in place */
                    g++;
                } else if (Ygj>pv) {
                    /* KO left : swap */
                    Yj[g] = Yj[h]; Yj[h] = Ygj;
                    Ij[g] = Ij[h]; Ij[h] = Igj;
                    h--;
                }
            
                /* compare right value to pivot */
                Yhj = Yj[h];
                Ihj = Ij[h];
                if (Yhj>pv) {
                    /* OK right: keep in place */
                    h--;
                } else if (Yhj<pv) {
                    /* KO right: swap */
                    Yj[h] = Yj[g]; Yj[g] = Yhj;
                    Ij[h] = Ij[g]; Ij[g] = Ihj;
                    g++;
                } else if (Ygj==pv) {
                    /* assert Ygh==pv && Yhj==pv */
                    if (g-lb<ub-h) g++; else h--;
                }
            }
            /* assert: g==h is the pivot location */
            
            if (M<g) {
                /* insert left chunk if the mid cell lies in it */
                t[0] =  lb; /* new lower bound */
                t[1] = g-1; /* new upper bound */
            } else if (g<M) {
                /* insert right chunk if the mid cell lies in it */
                t[0] = g+1; /* new lower bound */
                t[1] =  ub; /* new upper bound */
            }
            else {
                /* nothing more to do */
                break;
            }
        }
    }
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    long i, j, k, R, C, np2i;
    double *X, *Y, *I, *rR, *lR, cl2i;
    double *Ij, *rRj, *lRj;
    mxArray *tmp;
    
    /* check input and outputs */
    mxAssert(nrhs==1,"This function takes a single input arguments");
    mxAssert(nlhs>=1,"This function takes at least one output arguments");
    mxAssert(nlhs<=4,"This function takes no more than four output arguments");
    
    /* get pointer to data */
    X = mxGetPr(prhs[0]);
    
    /* get size of data */
    R = mxGetM(prhs[0]);
    C = mxGetN(prhs[0]);
    
    /* allocate first output */
    plhs[0] = mxCreateDoubleMatrix(R,C,mxREAL);
    Y = mxGetPr(plhs[0]);

    /* allocate indexes */
    tmp = mxCreateDoubleMatrix(R,C,mxREAL);
    I = mxGetPr(tmp);
    
#ifdef _OPENMP    
    omp_set_num_threads(16);
    /*mexPrintf("Number of threads = %i(%i)\n",omp_get_num_threads(),omp_get_max_threads());*/
#endif

    /* sort */
    minmedmax(R,C,X,Y,I);
    
    /* initialise second optional output (indexes) */
    if (nlhs>1) {
        plhs[1] = tmp;
    }
    
    /* initialise third optional output (median index) */
    if (nlhs>2) {
        plhs[2] = mxCreateDoubleScalar(ceil((double)R/2.0));
    }
}

