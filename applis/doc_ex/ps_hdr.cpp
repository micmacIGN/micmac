#include "general/all.h"



void PS(char * name)
{
cout << "DEB : " << name << "\n";
      char  buf[200];
      sprintf(buf,"../ARTICLE/HDR/EXPOSE/SYSECA/%s.tif",name);
      Tiff_Im Im(buf);

      Elise_Palette pal = Gray_Pal(256);
      if (Im.phot_interp() == Tiff_Im::RGBPalette)
         pal = Im.pal();
      else if (Im.NbBits() == 1)
         pal = Disc_Pal::P8COL();

     //  palette allocation

      Elise_Set_Of_Palette SOP(newl(pal));

     // Creation of postscript windows

      sprintf(buf,"../ARTICLE/HDR/EXPOSE/SYSECA/%s.eps",name);
    
      PS_Display disp(buf,"Mon beau fichier ps",SOP,false);
      PS_Window W =  disp.w_centered_max(Im.sz(),Pt2dr(4.0,4.0));

      Fonc_Num f0 = Im.in();
      if (Im.NbBits() ==  1)
          f0 = 1 - f0;
      ELISE_COPY
      (
          Im.all_pts(),
          f0,
          W.out(pal)
      );
cout << "FIN : " << name << "\n";
}



int  main(int,char **)
{

    PS("smal_toit");
    PS("small_cadastre");
    PS("small_droite");
    PS("small_gauche");

    return 0;
}
