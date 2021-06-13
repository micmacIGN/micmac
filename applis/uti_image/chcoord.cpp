/*Header-MicMac-eLiSe-25/06/2007

    MicMac : Multi Image Correspondances par Methodes Automatiques de Correlation
    eLiSe  : ELements of an Image Software Environnement

    www.micmac.ign.fr

   
    Copyright : Institut Geographique National
    Author : Marc Pierrot Deseilligny
    Contributors : Gregoire Maillet, Didier Boldo.

[1] M. Pierrot-Deseilligny, N. Paparoditis.
    "A multiresolution and optimization-based image matching approach:
    An application to surface reconstruction from SPOT5-HRS stereo imagery."
    In IAPRS vol XXXVI-1/W41 in ISPRS Workshop On Topographic Mapping From Space
    (With Special Emphasis on Small Satellites), Ankara, Turquie, 02-2006.

[2] M. Pierrot-Deseilligny, "MicMac, un lociel de mise en correspondance
    d'images, adapte au contexte geograhique" to appears in 
    Bulletin d'information de l'Institut Geographique National, 2007.

Francais :

   MicMac est un logiciel de mise en correspondance d'image adapte 
   au contexte de recherche en information geographique. Il s'appuie sur
   la bibliotheque de manipulation d'image eLiSe. Il est distibue sous la
   licences Cecill-B.  Voir en bas de fichier et  http://www.cecill.info.


English :

    MicMac is an open source software specialized in image matching
    for research in geographic information. MicMac is built on the
    eLiSe image library. MicMac is governed by the  "Cecill-B licence".
    See below and http://www.cecill.info.

Header-MicMac-eLiSe-25/06/2007*/
#include "general/all.h"
#include "private/all.h"
#include <algorithm>


Box2di Box(Flux_Pts flx,Fonc_Num f)
{
     Pt2dr p0,p1;
     ELISE_COPY(flx,Rconv(f),p0.VMin()|p1.VMax());
     return Box2di(round_down(p0), round_up(p1));
}

Box2di Box(Pt2di p0,Pt2di p1,Fonc_Num f)
{
   return Box(border_rect(p0,p1,3),f);
}

Box2di Box(Pt2di p1,Fonc_Num f)
{
    return Box(Pt2di(0,0),p1,f);
}


class  BufIm
{
      public :
          Fonc_Num in() {return _In;}
          Output out()  {return _Out;}
          void SetSize(Pt2di);
          BufIm(INT dim,INT def);

      private :

          INT _dim;
          INT _def;
          Pt2di _sz;
          Fonc_Num _In;
          Output   _Out;
};


BufIm::BufIm(INT dim,INT def) :
    _dim (dim),
    _def (def),
    _sz  (-1,-1),
    _In  (0),
    _Out (Output::onul())
{
}

void BufIm::SetSize(Pt2di sz)
{
    if (sz.xety_inf_ou_egal(_sz))
       return;

    _sz = sz;
    Im2D_U_INT1 i0(_sz.x,_sz.y);
    _In =  i0.in(_def);
    _Out = i0.oclip();

    for (INT d=1; d<_dim; d++)
    {
        Im2D_U_INT1 i(_sz.x,_sz.y);
        _In = Virgule( _In,i.in(_def));
        _Out = Virgule(_Out,i.oclip());
    }
}

void  ChangCoord
      (
           INT                      MoBuf,
           Fonc_Num                 I2O_init,
           Fonc_Num                 O2I_init,
           string                   Name,
           string                   NameOut,
           Pt2di                    Dalles,
           INT                      NbBits,
           Tiff_Im::COMPR_TYPE      MC,
           INT                      Def,
           bool                     PPV,
           bool                     Visu
       )
{
    REAL OctBuf = MoBuf * 1e6;
    Pt2di  BufDalles = Pt2di(Pt2dr(sqrt(OctBuf),sqrt(OctBuf)));
    Tiff_Im TIFF = Tiff_Im::StdConv(Name); 
    BufDalles =arrondi_sup(BufDalles,Dalles);
    Box2di bOut = Box(TIFF.sz(),I2O_init);

    Fonc_Num I2O_Ok = I2O_init-Fonc_Num(bOut._p0.x);
    Fonc_Num O2I_Ok = O2I_init[Virgule(FX+bOut._p0.x,FY+bOut._p0.y)];

    Pt2di SzOut = bOut.sz();


    BufIm Bim(TIFF.NbChannel(),Def);

    GenIm::type_el TypeOut =  type_im
                              (
                                  TIFF.IntegralType(),
                                  NbBits,
                                  TIFF.SigneType(),
                                  true
                              );
    Tiff_Im TOUT =
         (TIFF.phot_interp() == Tiff_Im::RGBPalette)                  ?
         Tiff_Im(NameOut.c_str(),SzOut,TypeOut,MC,TIFF.pal())         : 
         Tiff_Im(NameOut.c_str(),SzOut,TypeOut,MC,TIFF.phot_interp()) ; 


    cout << "Sz Out = " << SzOut << "\n";
    Pt2di PRab (5,5);

    for (INT X0_o = 0; X0_o  < TOUT.sz().x ; X0_o +=BufDalles.x)
    {
        INT Tx_o = ElMin(BufDalles.x,TOUT.sz().x-X0_o);
        for (INT Y0_o = 0; Y0_o  < TOUT.sz().y ; Y0_o +=BufDalles.y)
        {
            INT Ty_o = ElMin(BufDalles.y,TOUT.sz().y-Y0_o);
            Pt2di P0dOut(X0_o,Y0_o);
            Pt2di SzdOut(Tx_o,Ty_o);
            Pt2di P1dOut = P0dOut+SzdOut;

            cout  << "   Write rect = " << P0dOut  << " " << P1dOut << "\n";
           
            Box2di BoxdIn  =  Box(P0dOut-PRab,P1dOut+PRab,O2I_Ok);
            Pt2di SzdIn = BoxdIn.sz();
            Bim.SetSize(SzdIn);
            ELISE_COPY
            (
		 rectangle(Pt2di(0,0),SzdIn),
                 trans(TIFF.in(Def),BoxdIn._p0),
                 Bim.out()
            );


            Fonc_Num O2Iloc = O2I_Ok -Fonc_Num(BoxdIn._p0);
            if (PPV) 
               O2Iloc = Iconv(O2Iloc);

            ELISE_COPY
            (
                 rectangle(P0dOut,P1dOut),
                 Bim.in()[O2Iloc],
                 TOUT.out()
            );
        }
    }
}


int main(int argc,char ** argv)
{
    string Name,NameOut;
    Pt2di  Dalles(256,256);
    INT    MoBuf = 1;


    INT NbBits = -1;
    string Compr("Id"); // Par defaut on reprend le meme mode de compr.
    INT  Visu = 0;
    INT  Def = 0;
    INT  PPV = 1;
    REAL teta = 0.0;

    ElInitArgMain
    (
        argc,argv,
        LArgMain()  << EAM(Name) 
                    << EAM(NameOut) ,
        LArgMain()  << EAM(Dalles,"Dalles",true)
                    << EAM(MoBuf,"MoBuf",true)
                    << EAM(NbBits,"NbBits",true)
                    << EAM(Compr,"Compr",true)
                    << EAM(Visu,"Visu",true)
                    << EAM(Def,"Def",true)
                    << EAM(PPV,"PPV",true)
                    << EAM(teta,"teta",true)
			
    );	

    teta = teta * PI/180.0;
    REAL ct = cos(teta);
    REAL st = sin(teta);



    Tiff_Im TIFF = Tiff_Im::StdConv(Name); 
    if (NbBits==-1)
       NbBits = TIFF.NbBits();


    Tiff_Im::COMPR_TYPE  MC = TIFF.mode_compr();
    if (Compr != "Id")
       MC = Tiff_Im::mode_compr(Compr);

    Fonc_Num I2O_init = Virgule(ct*FX+st*FY,-st*FX+ct*FY);
    Fonc_Num O2I_init = Virgule(ct*FX-st*FY,+st*FX+ct*FY);


    ChangCoord
    (
        MoBuf,
        I2O_init,
        O2I_init,
        Name,
        NameOut,
        Dalles,
        NbBits,
        MC,
        Def,
        PPV,
        Visu
    );

    return 0;
}







/*Footer-MicMac-eLiSe-25/06/2007

Ce logiciel est un programme informatique servant à la mise en
correspondances d'images pour la reconstruction du relief.

Ce logiciel est régi par la licence CeCILL-B soumise au droit français et
respectant les principes de diffusion des logiciels libres. Vous pouvez
utiliser, modifier et/ou redistribuer ce programme sous les conditions
de la licence CeCILL-B telle que diffusée par le CEA, le CNRS et l'INRIA 
sur le site "http://www.cecill.info".

En contrepartie de l'accessibilité au code source et des droits de copie,
de modification et de redistribution accordés par cette licence, il n'est
offert aux utilisateurs qu'une garantie limitée.  Pour les mêmes raisons,
seule une responsabilité restreinte pèse sur l'auteur du programme,  le
titulaire des droits patrimoniaux et les concédants successifs.

A cet égard  l'attention de l'utilisateur est attirée sur les risques
associés au chargement,  à l'utilisation,  à la modification et/ou au
développement et à la reproduction du logiciel par l'utilisateur étant 
donné sa spécificité de logiciel libre, qui peut le rendre complexe à 
manipuler et qui le réserve donc à des développeurs et des professionnels
avertis possédant  des  connaissances  informatiques approfondies.  Les
utilisateurs sont donc invités à charger  et  tester  l'adéquation  du
logiciel à leurs besoins dans des conditions permettant d'assurer la
sécurité de leurs systèmes et ou de leurs données et, plus généralement, 
à l'utiliser et l'exploiter dans les mêmes conditions de sécurité. 

Le fait que vous puissiez accéder à cet en-tête signifie que vous avez 
pris connaissance de la licence CeCILL-B, et que vous en avez accepté les
termes.
Footer-MicMac-eLiSe-25/06/2007*/
