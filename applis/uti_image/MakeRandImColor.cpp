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



class OneImageColor
{
    public :
       OneImageColor(INT zoom,Pt2di sz);
       void SetColor(INT Col,ElList<Pt2di> Lpt0);
       void write(const std::string & name,Disc_Pal aPal);

    private :

       INT                mZoom;
       Pt2di              mSzIm;
       PackB_IM<U_INT1>   mPckIm;
};


void OneImageColor::write(const std::string & name,Disc_Pal aPal)
{
    char bufZ[6];
    sprintf(bufZ,"%d",mZoom);

    string aFullName = name + string("Reduc") + bufZ  + ".tif";

    Tiff_Im aTif
            (
                 aFullName.c_str(),
                 mSzIm,
                 GenIm::u_int1,
                 Tiff_Im::PackBits_Compr,
                 aPal,
                     L_Arg_Opt_Tiff() 
                 + Arg_Tiff(Tiff_Im::ATiles(Pt2di(256,256)))
            );

    ELISE_COPY(aTif.all_pts(),mPckIm.in(),aTif.out());
}




OneImageColor::OneImageColor
(
    INT   aZoom,
    Pt2di aSz
)  :
   mZoom  (aZoom),
   mSzIm  ((aSz+Pt2di(aZoom-1,aZoom-1))/aZoom),
   mPckIm (mSzIm.x,mSzIm.y,0,128)
{
}


void OneImageColor::SetColor(INT Col,ElList<Pt2di> Lpt0)
{
    ElList<Pt2di> LptLoc;

    for (;!Lpt0.empty() ; Lpt0 = Lpt0.cdr())
        LptLoc = LptLoc + (Lpt0.car()/mZoom);

    ELISE_COPY(polygone(LptLoc),Col,mPckIm.out());
}


class PyrImColor
{
    public :

       PyrImColor
       (
            const std::string & aName,
            const std::vector<INT> & Zooms,
            Pt2di            aSz,
            INT              aNbCol
       );

       INT AddASurf(INT k);
       void write();
       
       

    private :

      
      INT SetColor(INT Col,ElList<Pt2di> Lpt0);
      Pt2di   PRandAbs();
      Pt2di   PRandAround(Pt2di,REAL Dist);

      std::string                   mName;
      INT                           mNbCol;
      Elise_colour *                mCols;
      Disc_Pal                      mPal;
      std::vector<OneImageColor *>  mZIm;
      Pt2di                         mSz;
      REAL                          mRedW;
      Pt2di                         mSzW;
      Video_Win                     mW;


      static const Pt2dr  theSzMax;

};


void PyrImColor::write()
{
    for (INT k=0; k<(INT) mZIm.size() ; k++)
       mZIm[k]->write(mName,mPal);
    
}

Pt2di   PyrImColor::PRandAbs()
{
    return Pt2di(mSz.x*NRrandom3(),mSz.y*NRrandom3());
}

Pt2di   PyrImColor::PRandAround(Pt2di PBase,REAL Dist)
{
   return Sup
          (
             Pt2di(0,0),
             Inf
             (
                 mSz-Pt2di(1,1),
                    PBase
                 + Pt2di(Dist*(NRrandom3()-0.5)*2,Dist*(NRrandom3()-0.5)*2)
             )
          );
}

const Pt2dr PyrImColor::theSzMax(800,600);


INT PyrImColor::SetColor(INT Col,ElList<Pt2di> Lpt0)
{
    Col = ElMin(Col,mNbCol-1);
    for (INT k=0; k<(INT) mZIm.size() ; k++)
       mZIm[k]->SetColor(Col,Lpt0);

    INT res;
    ELISE_COPY
    (
         polygone(Lpt0),
         Col,
         (((mNbCol<=8) ? mW.odisc(): mW.ogray())|(sigma(res)<<1))
    )
;
    return res;
}


INT PyrImColor::AddASurf(INT Cpt)
{
    INT NbPts = (INT)(3 + ElSquare(NRrandom3()) * 10);

    REAL Dist = 30 + pow(NRrandom3(),3.0) * 300;

    Pt2di p0 = PRandAbs();

    ElList<Pt2di> pts;

    for (INT k=0 ; k<NbPts ; k++)
       pts = pts + (PRandAround(p0,Dist));

    return SetColor(1+(INT)(NRrandom3() * (mNbCol-1)),pts);
    
}


Elise_colour * ATabCol(INT aNb)
{
   Elise_colour * res = new Elise_colour [aNb];
   for (INT c=0; c<aNb; c++)
       res[c] = Elise_colour::rand();

   return res;
}


PyrImColor::PyrImColor
(
   const std::string &      aName,
   const std::vector<INT> & Zooms,
   Pt2di            aSz,
   INT              aNbCol
)  :
   mName  (aName),
   mNbCol (aNbCol),
   mCols  (ATabCol(256)),
   mPal   (mCols,256),
   mZIm   (),
   mSz    (aSz),
   mRedW  (ElMin(theSzMax.x/aSz.x,theSzMax.y/aSz.y)),
   mSzW   (Pt2dr(mSz)*mRedW),
   mW     (Video_Win::WStd(mSz,mRedW))
{
    for (INT k=0; k<(INT) Zooms.size() ; k++)
        mZIm.push_back(new OneImageColor(Zooms[k],aSz));
}



int main(int argc,char ** argv)
{

    INT NbCol = 8;
    Pt2di SzIm (4000,4000);
    REAL CoeffRempl = 1.0;

	string Name;
    std::vector<INT> Zooms;
    for (INT k=0 ; k<6 ; k++)
         Zooms.push_back(1<<k);

	ElInitArgMain
	(
		argc,argv,
		LArgMain() 	<< EAM(Name) ,
		LArgMain()  << EAM(NbCol,"NbCol",true)
                    << EAM(SzIm,"SzIm",true)
                    << EAM(CoeffRempl,"CoeffRempl",true)
	);	


    PyrImColor aPyr(Name,Zooms,SzIm,NbCol);


    double SCible = SzIm.x*SzIm.y * CoeffRempl;
    double res =0.0;
    INT k=0;
    while (res < SCible)
    {
        k++;
        res += aPyr.AddASurf(k);
        cout << k <<" " << res/SCible <<  "\n";
    }

    aPyr.write();

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
