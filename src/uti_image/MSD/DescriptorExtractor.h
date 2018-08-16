#ifndef DESCRIPTOR_EXTRACTOR_H_
#define  DESCRIPTOR_EXTRACTOR_H_

#include "../../uti_image/Digeo/Digeo.h"
#include "../../uti_image/Digeo/DigeoPoint.h"


template <class tData, class tComp>
class DescriptorExtractor
{
    public:
        DescriptorExtractor(Im2D<tData, tComp> Image);

        ~DescriptorExtractor();

        void gradient(REAL8 i_maxValue);
        // o_descritpor must be of size DIGEO_DESCRIPTOR_SIZE
        void describe(REAL8 i_x, REAL8 i_y, REAL8 i_localScale, REAL8 i_angle, REAL8 *o_descriptor );
        void normalizeDescriptor( REAL8 *io_descriptor );
        void truncateDescriptor( REAL8 *io_descriptor );
        void normalize_and_truncate( REAL8 *io_descriptor );

    protected:
        Im2D<tData, tComp> m_image;
        Im2D<REAL4, REAL8> m_gradim;
};
#endif
