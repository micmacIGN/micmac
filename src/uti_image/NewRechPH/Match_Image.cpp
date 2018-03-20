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


#include "NewRechPH.h"

class cAFM_Im;
class cAFM_Im_Master;
class cAFM_Im_Sec ;
class cAppli_FitsMatch1Im;

//================================


class cAFM_Im
{
     public :
         cAFM_Im (const std::string  &,cAppli_FitsMatch1Im &);
         ~cAFM_Im ();
     protected :
         cAppli_FitsMatch1Im & mAppli;
         std::string mNameIm;
         cSetPCarac                             mSetPC;
         std::vector<std::vector<cCompileOPC> > mVOPC;
};

class cAFM_Im_Master : cAFM_Im
{
     public :
         cAFM_Im_Master (const std::string  &,cAppli_FitsMatch1Im &);
};


class cAFM_Im_Sec : cAFM_Im
{
     public :
         cAFM_Im_Sec (const std::string  &,cAppli_FitsMatch1Im &);
};
         


class cAppli_FitsMatch1Im
{
     public :
          cAppli_FitsMatch1Im(int argc,char ** argv);
          const std::string &   ExtNewH () const {return    mExtNewH;}
          const cFitsParam & FitsPm() const {return mFitsPm;}
     private :

          cFitsParam         mFitsPm;
          std::string        mNameMaster;
          std::string        mPatIm;
          cElemAppliSetFile  mEASF;
          cAFM_Im_Master *   mImMast;
          cAFM_Im_Sec *      mCurImSec;
          std::string        mNameXmlFits;
          std::string        mExtNewH;
};

/*************************************************/
/*                                               */
/*             ::                                */
/*                                               */
/*************************************************/

void InitOneLabelFitsPm(cFitsOneBin & aFB,const std::string & aDir,eTypePtRemark aLab)
{
    std::string aName = aDir + aFB.PrefName() +  eToString(aLab) +  aFB.PostName().Val();


    cCompCB aCB = StdGetFromNRPH(aName,CompCB);
    aFB.CCB().SetVal(aCB);

    // std::cout <<   aCB.CompCBOneBit().size()  << "  "  << aName << "\n";
}

void InitOneFitsPm(cFitsOneLabel & aFOL,const std::string & aDir,eTypePtRemark aLab)
{
    InitOneLabelFitsPm(aFOL.BinIndexed(),aDir,aLab);
    ELISE_ASSERT(aFOL.BinIndexed().CCB().Val().CompCBOneBit().size()<=16,"InitOneFitsPm");
    InitOneLabelFitsPm(aFOL.BinDecision(),aDir,aLab);
}

void InitFitsPm(cFitsParam & aFP,const std::string & aDir, const std::string & aName)
{
    aFP = StdGetFromNRPH(aDir+aName,FitsParam);
    InitOneFitsPm(aFP.OverLap(),aDir,aFP.KindOl());
}

/*************************************************/
/*                                               */
/*           cAFM_Im                             */
/*                                               */
/*************************************************/

bool CmpCPOC(const cCompileOPC &aP1,const  cCompileOPC &aP2)
{
   return aP1.mOPC.ScaleStab() > aP2.mOPC.ScaleStab();
}

cAFM_Im::cAFM_Im (const std::string  & aNameIm,cAppli_FitsMatch1Im & anAppli) :
   mAppli  (anAppli),
   mNameIm (aNameIm),
   mVOPC   (int(eTIR_NoLabel))
{
    std::string aNamePC = NameFileNewPCarac(mNameIm,true,anAppli.ExtNewH());

    mSetPC = StdGetFromNRPH(aNamePC,SetPCarac);
    std::cout << "cAFM_Im::cAFM " << aNameIm << " " << mSetPC.OnePCarac().size() << "\n";

    for (const auto & aPC : mSetPC.OnePCarac())
    {
        mVOPC.at(int(aPC.Kind())).push_back(cCompileOPC(aPC));
    }

    const cFitsParam & aFitsPM = anAppli.FitsPm();
    const cFitsOneLabel & aFOL = aFitsPM.OverLap();
    
}

cAFM_Im::~cAFM_Im()
{
   static cSetPCarac TheSetPC;
   mSetPC = TheSetPC;
}

/*************************************************/
/*                                               */
/*           cAFM_Im_Master                      */
/*                                               */
/*************************************************/

cAFM_Im_Master::cAFM_Im_Master(const std::string  & aName,cAppli_FitsMatch1Im & anApli) :
    cAFM_Im(aName,anApli)
{
}

/*************************************************/
/*                                               */
/*           cAFM_Im_Sec                         */
/*                                               */
/*************************************************/

cAFM_Im_Sec::cAFM_Im_Sec(const std::string  & aName,cAppli_FitsMatch1Im & anApli) :
    cAFM_Im(aName,anApli)
{
}

/*************************************************/
/*                                               */
/*           cAppli_FitsMatch1Im                 */
/*                                               */
/*************************************************/

std::string TheDirXmlFits=    string("include")    + ELISE_CAR_DIR
                            + string("XML_MicMac") + ELISE_CAR_DIR 
                            + string("Fits")       + ELISE_CAR_DIR;


cAppli_FitsMatch1Im::cAppli_FitsMatch1Im(int argc,char ** argv) :
   mImMast      (nullptr),
   mCurImSec    (nullptr),
   mNameXmlFits ("FitsParam.xml"),
   mExtNewH   ("")
{
   ElInitArgMain
   (
         argc,argv,
         LArgMain()  << EAMC(mNameMaster, "First Image")
                     << EAMC(mPatIm, "Name Image2"),
         LArgMain()  << EAM(mNameXmlFits,"XmlFits",true,"Name of xml file for Fits parameters")
                     <<  EAM(mExtNewH,"ExtPC",true,"Extension for P cararc to NewPH... ")
   );
   
   InitFitsPm(mFitsPm,MMDir()+TheDirXmlFits,mNameXmlFits);
   // MMDir()
   // FitsParam.xml
   mEASF.Init(mPatIm);


   mImMast = new cAFM_Im_Master(mNameMaster,*this);

   for (const auto &  aName : *(mEASF.SetIm()))
   {
       if (aName != mNameMaster)
       {
           mCurImSec = new  cAFM_Im_Sec(aName,*this);

           delete mCurImSec;
            mCurImSec = nullptr;
       }
   }
}


int CPP_FitsMatch1Im(int argc,char ** argv)
{
   cAppli_FitsMatch1Im anAppli(argc,argv);
   return EXIT_SUCCESS;
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
aooter-MicMac-eLiSe-25/06/2007*/
