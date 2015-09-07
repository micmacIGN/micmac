#include "general/all.h"

int  main(int,char **)
{
     Tiff_Im Lena("../IM_ELISE/TIFF/lena.tif");
     Im2D_U_INT2 I (256,256,0);

     ELISE_COPY
     (
        rectangle(Pt2di(0,0),Lena.sz()),
        Lena.in(),
        I.histo().chc((FX,FY)/2)
     );
     ELISE_COPY(I.all_pts(),I.in()/4,I.out());

      Tiff_Im Fres
              (
                 "DOC/mini_lena.tif",
                 Pt2di(256,256),
                 GenIm::u_int1,
                 Tiff_Im::LZW_Compr,
                 Tiff_Im::BlackIsZero
              );
     ELISE_COPY(I.all_pts(),(I.in()/1)*1,Fres.out());
}


