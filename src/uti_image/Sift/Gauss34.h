#ifndef __GAUSS34__
#define __GAUSS34__

// this is from Andrea Vedaldi's siftpp (now VLFeat)

inline bool siftpp__gauss33_invert_b( Real_ *i_m, Real_ *i_b )
{
    #define at(i,j)     (i_m[(i)+(j)*4])
    // Gauss elimination
    for(int j = 0 ; j < 3 ; ++j)
    {
        // look for leading pivot
        Real_ maxa = 0;
        Real_ maxabsa = 0;
        int maxi = -1;
        int i;
        for( i=j; i<3; i++ )
        {
            Real_ a    = at(i,j);
            Real_ absa = fabsf( a );
            if ( absa>maxabsa )
            {
                maxa    = a ;
                maxabsa = absa ;
                maxi    = i ;
            }
        }

        // singular?
        if ( maxabsa<1e-10 )
        {
            i_m[3]  = 0 ;
            i_m[7]  = 0 ;
            i_m[11] = 0 ;
            return false;
        }

        i = maxi ;

        // swap j-th row with i-th row and
        // normalize j-th row
        for ( int jj=j; jj<3; jj++ ){
            std::swap( at(j,jj) , at(i,jj) ) ;
            at(j,jj) /= maxa ;
        }
        std::swap( i_b[j], i_b[i] ) ;
        i_b[j] /= maxa ;
        
        // elimination
        for ( int ii=j+1; ii<3; ii++ )
        {
            Real_ x = at(ii,j) ;
            for( int jj=j; jj<3; jj++ )
					at(ii,jj) -= x*at(j,jj);
            i_b[ii] -= x*i_b[j] ;
        }
    }

    // backward substitution
    for ( int i=2; i>0; i-- )
    {
        Real_ x = i_b[i];
        for( int ii=i-1; ii>=0; ii-- )
            i_b[ii] -= x * at(ii,i);
    }
    return true;
}

#endif
