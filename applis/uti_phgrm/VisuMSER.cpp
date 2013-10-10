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


void Banniere_MSER()
{
   std::cout << "\n";
   std::cout <<  " *********************************\n";
   std::cout <<  " *     VISU MSER                 *\n";
   std::cout <<  " *********************************\n";
}

class cMSER_Ell
{
     public :
         Pt2dr mC;
         double mX2,mXY,mY2;

        static std::vector<cMSER_Ell> FromMSER(const std::string & aName);
};

#define aSzBUF 500
std::vector<cMSER_Ell> cMSER_Ell::FromMSER(const std::string & aName)
{
  std::vector<cMSER_Ell> aRes;
  ifstream    aFile;
  aFile.open(aName.c_str(),ios::in);

  double aV;
  int aNbEl;

  aFile >> aV >> aNbEl;

  std::cout << "V= " << aV << " Nb=" << aNbEl << "\n";
  
  for ( int aK=0 ; aK<aNbEl ; aK++)
  {
     cMSER_Ell anE;
     aFile >> anE.mC.x >>  anE.mC.y >> anE.mX2 >> anE.mXY >> anE.mY2 ;
     aRes.push_back(anE);
  }


  aFile.close();
  return aRes;
}



class cSubW_VMser
{
    public :
        cSubW_VMser
	(
	    Video_Win           aWPos,
            Video_Win::ePosRel  aPR,
	    const std::string&  aFile,
	    Pt2di               aSz,
            int                 aNumP
       );
        Video_Win & W() {return  mW;}
	void SetDec(Pt2dr,double aZ );
    private :
        Video_Win   mW;
	Tiff_Im     mTif;
        std::vector<cMSER_Ell> mVEl;
};

void cSubW_VMser::SetDec(Pt2dr aDec,double aZ )
{
    ELISE_COPY
    (
         mW.all_pts(),
         StdFoncChScale
         (
              mTif.in(0),
              aDec,
              Pt2dr(aZ,aZ),
              Pt2dr(1,1)
         ),
          mW.ogray()
    );

    Video_Win aWZ =  mW.chc(aDec,Pt2dr(1/aZ,1/aZ));

    for (int aK= 0 ; aK<int(mVEl.size()) ; aK++)
    {
         aWZ.draw_circle_loc(mVEl[aK].mC,2.0,aWZ.pdisc()(P8COL::red));
    }
}

cSubW_VMser::cSubW_VMser
(
      Video_Win           aWPos,
      Video_Win::ePosRel  aPR,
      const std::string&  aFile,
      Pt2di               aSz,
      int                 aNumP
)  :
    mW   (aWPos,aPR,aSz),
    mTif (Tiff_Im::StdConv(aFile))
{
   ELISE_COPY(mW.all_pts(),((FX+2*FY)*10)%256,mW.ogray());

   std::string  aNMSER = StdPrefix(aFile) + ".mser-" + ToString(aNumP);

   if (aNumP==2)
   {
      mVEl = cMSER_Ell::FromMSER(aNMSER);
   }
}


class cAppli_Visu_MSER
{
    public :
        cAppli_Visu_MSER(int argc,char ** argv);

	void NoOp(){}
    private :
        std::string KCh(int aK) const
        {
            return mDir + mChans.c_str()[aK] + ".tif";
        }
        void AddSubw ( const std::string&  aFile);

        std::vector<cSubW_VMser*> mSubW;
        Pt2di                  mSzSubW;
        Pt2di       mSzWP;

	// Tiff_Im *           mT1;
	Video_Win*          mW;

        std::string         mDir;
        std::string         mChans;
        int                 mModeMSER;
};

void cAppli_Visu_MSER::AddSubw(const std::string&  aFile)
{
 
    int aPer = 2;
    Video_Win * aWPos =  0;
    Video_Win::ePosRel aPR = Video_Win::eSamePos;

    
    if ((mSubW.size()%aPer)==0 )
    {
        if (mSubW.size()==0)
        {
            aWPos = mW;
            aPR = Video_Win::eSamePos;
        }
        else
        {
            aWPos = &(mSubW[mSubW.size()-aPer]->W());
            aPR = Video_Win::eBasG ;
        }
    }
    else
    {
       aWPos = & (mSubW.back()->W());
       aPR =  Video_Win::eDroiteH;
    }
   

   // mSubW.empty() ? *mW : mSubW.back()->W();

    // Video_Win::ePosRel aPR =  Video_Win::eSamePos;
    // if (mSubW.size() != 0)
      // aPR = ((mSubW.size()%2)==0) ? Video_Win::eBasG :  Video_Win::eDroiteH;


    std::cout << mSubW.size() << " " << aPR << "\n";


    cSubW_VMser * aSub = new cSubW_VMser ( *aWPos, aPR, aFile, mSzSubW,mModeMSER);
    mSubW.push_back(aSub);
}


cAppli_Visu_MSER::cAppli_Visu_MSER(int argc,char ** argv)
{
    mSzWP   = Pt2di(700,500);
    mSzSubW = Pt2di(400,400);


    std::string Toto;

    ElInitArgMain
    (
           argc,argv,
           LArgMain() << EAM(mDir) 
                      << EAM(mChans) 
                      << EAM(mModeMSER) ,
           LArgMain() << EAM(Toto,"toto",true)
    );

    // std::string aNTif1 = KCh(0);

    // mT1  = new Tiff_Im(Tiff_Im::StdConv(aNTif1));
    mW =  Video_Win::PtrWStd(mSzWP);

    for (int aK=0 ; aK<int(mChans.size()) ; aK++)
        AddSubw(KCh(aK));

    while (1)
    {
        double x,y,z;
        cin >> x >> y >> z;
        for (int aKW=0 ; aKW<int(mSubW.size()) ; aKW++)
        {
            mSubW[aKW]->SetDec(Pt2dr(x,y),z);
        }
    }
}


int main(int argc,char ** argv)
{
    cAppli_Visu_MSER aAP(argc,argv);
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
