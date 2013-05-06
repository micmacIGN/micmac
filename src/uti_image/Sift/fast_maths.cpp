#include "Sift.h"

namespace fast_maths{
    const int  expnTableSize = 256;
    const Real_ expnTableMax  = Real_( 25.0 );
          Real_ expnTable[expnTableSize+1];

    struct buildExpnTable
    {
        buildExpnTable()
        {
            for ( int k=0; k<expnTableSize+1; k++ )
                expnTable[k] = exp( -Real_(k)/expnTableSize*expnTableMax );
        }
    } _buildExpnTable;
}
