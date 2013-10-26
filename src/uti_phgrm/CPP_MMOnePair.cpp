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
#include "StdAfx.h"


class cAppliMMOnePair;  // Appli principale
class cMMOnePair;  // Cree pour gerer l'ordre d'initialisatrion (a faire avant cAppliWithSetImage)


std::string Pair2PattWithoutDir(std::string & aDirRes,const std::string & aName1,const std::string & aName2);
std::string Name2PattWithoutDir(std::string & aDirRes,const std::vector<std::string>  & aVName1);

class cMMOnePair 
{
    public :
      cMMOnePair(int argc,char ** argv);
    protected :

      int              mZoom0;
      int              mZoomF;
      bool             mByEpip;
      bool             mCMS;
      bool             mForceCreateE;
      std::string      mNameIm1Init;
      std::string      mNameIm2Init;
      std::string      mNameOriInit;

      std::string      mNameIm1;
      std::string      mNameIm2;
      std::string      mNameOri;
      cCpleEpip        *mCpleE;
      bool             mDoubleSens;
      bool             mNoOri;


      std::string      mDirP;
      std::string      mPatP;
      std::vector<char *>        mArgcAWS;  // AppliWithSetImage
};

class cAppliMMOnePair : public cMMOnePair,
                        public cAppliWithSetImage
{
     public :
         cAppliMMOnePair(int argc,char ** argv);
     private :
         void MatchOneWay(bool MasterIs1);
         cImaMM * mIm1;
         cImaMM * mIm2;
};

/*****************************************************************/
/*                                                               */
/*             cMMOnePair                                        */
/*                                                               */
/*****************************************************************/



cMMOnePair::cMMOnePair(int argc,char ** argv) :
    mZoom0        (64),
    mZoomF        (2),
    mByEpip       (true),
    mCMS          (true),
    mForceCreateE (false),
    mCpleE        (0),
    mDoubleSens   (true),
    mNoOri        (false)
{
  ElInitArgMain
  (
        argc,argv,
        LArgMain()  << EAMC(mNameIm1Init,"Name Im1")
                    << EAMC(mNameIm2Init,"Name Im2")
                    << EAMC(mNameOriInit,"Orientation (if NONE, work directly on epipolar)"),
        LArgMain()  << EAM(mZoom0,"Zoom0",true,"Zoom Init, Def=64")
                    << EAM(mZoomF,"ZoomF",true,"Zoom Final, Def=1")
                    << EAM(mByEpip,"CreateE",true," Create Epipolar (def = true when appliable)")
                    << EAM(mDoubleSens,"2Way",true,"Match in 2 Way (Def=true)")
                    << EAM(mCMS,"CMS",true,"Multi Scale Coreel (Def=ByEpip)")
  );
  mNoOri = (mNameOriInit=="NONE");
  if ((! EAMIsInit(&mByEpip)) && mNoOri)
     mByEpip = false;
  mDirP =DirOfFile(mNameIm1Init);

  if (!EAMIsInit(&mCMS)) 
     mCMS = mByEpip;

  if (mByEpip)
  {
       mCpleE = StdCpleEpip(mDirP,mNameOriInit,mNameIm1Init,mNameIm2Init);
       mNameIm1 =  mCpleE->LocNameImEpi(mNameIm1Init);
       mNameIm2 =  mCpleE->LocNameImEpi(mNameIm2Init);
       mNameOri =  "Epi";
       if (
               (! ELISE_fp::exist_file(mDirP+mNameIm1))
            || (! ELISE_fp::exist_file(mDirP+mNameIm2))
            || mForceCreateE
          )
       {
             std::string aCom =        MMBinFile(MM3DStr) 
                               + std::string(" CreateEpip ")
                               + " " + mNameIm1Init 
                               + " " + mNameIm2Init 
                               + " " + mNameOriInit;
             System(aCom);
       }
  }
  else
  {
       mNameIm1 = mNameIm1Init;
       mNameIm2 = mNameIm2Init;
       mNameOri = mNameOriInit;
  }

  mPatP = Pair2PattWithoutDir(mDirP,mNameIm1,mNameIm2);
  mPatP = mDirP+mPatP;
  mArgcAWS.push_back(const_cast<char *>(mPatP.c_str()));
  mArgcAWS.push_back(const_cast<char *>(mNameOri.c_str()));
}

/*****************************************************************/
/*                                                               */
/*             cAppliMMOnePair                                   */
/*                                                               */
/*****************************************************************/

cAppliMMOnePair::cAppliMMOnePair(int argc,char ** argv) :
   cMMOnePair(argc,argv),
   cAppliWithSetImage(2,&(mArgcAWS[0]),0)
{
    ELISE_ASSERT(mImages.size()==2,"Expect exaclty 2 images in cAppliMMOnePair");
    mIm1 = mImages[0];
    mIm2 = mImages[1];

    MatchOneWay(true);
    if (mDoubleSens)
        MatchOneWay(false);
}

void cAppliMMOnePair::MatchOneWay(bool MasterIs1)
{
     std::string aNamA = MasterIs1 ? mNameIm1 : mNameIm2;
     std::string aNamB = MasterIs1 ? mNameIm2 : mNameIm1;

     std::string aCom =     MMBinFile(MM3DStr) 
                          + std::string(" MICMAC ")
                          +  XML_MM_File("MM-Epip.xml ") 
                          + " WorkDir="  + mDir
                          + " +Im1="     + aNamA   
                          + " +Im2="     + aNamB 
                          + " +ZoomF="   + ToString(mZoomF)
                          + " +Ori="     + mNameOri
                          + " +DoEpi="   + ToString(mByEpip || mNoOri)
                          + " +CMS="     + ToString(mCMS)
                      ;

     std::cout << aCom << "\n";
}

/*****************************************************************/
/*                                                               */
/*                ::                                             */
/*                                                               */
/*****************************************************************/

std::string Name2PattWithoutDir(std::string & aDirRes,const std::vector<std::string>  & aVName)
{
   ELISE_ASSERT(aVName.size()!=0,"Name2Patt");
   std::string aRes = "(";

   for (int aK=0 ; aK<int(aVName.size()) ; aK++)
   {
        std::string aDir,aName;
        SplitDirAndFile(aDir,aName,aVName[aK]);
        if (aK==0)
        {
            aDirRes = aDir;
        }
        else
        {
            ELISE_ASSERT(aDirRes == aDir,"Variable dir in Name2PattWithoutDir");
            aRes += "|";
        }
        aRes += "("  + aName + ")";
   }
   return aRes + ")";
}

std::string Pair2PattWithoutDir(std::string & aDir,const std::string & aName1,const std::string & aName2)
{
    std::vector<std::string> aVName;
    aVName.push_back(aName1);
    aVName.push_back(aName2);
    return Name2PattWithoutDir(aDir,aVName);
}

int MMOnePair_main(int argc,char ** argv)
{
    // for (int aK=0 ; aK<argc ; aK++) std::cout <<  "**[" <<  argv[aK] << "]\n";


   cAppliMMOnePair aMOP(argc,argv);

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
