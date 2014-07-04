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
#include "XML_GEN/all.h"

using namespace NS_ParamChantierPhotogram;

/*********************************************/
/*                                           */
/*                ::                         */
/*                                           */
/*********************************************/


cOrientationConique StdGetOC(const std::string & aFc)
{
   return StdGetObjFromFile<cOrientationConique>
          (
	     aFc,
             "include/XML_GEN/ParamChantierPhotogram.xml",
             "OrientationConique",
             "OrientationConique"
          );
}

void Banniere_Porto()
{
   std::cout << "\n";
   std::cout <<  " *********************************\n";
   std::cout <<  " *     P-hotogrammetrie,          *\n";
   std::cout <<  " *     O-rientation               *\n";
   std::cout <<  " *     R-elative pour             *\n";
   std::cout <<  " *     T-ie-points                *\n";
   std::cout <<  " *     O-tomatiks                 *\n";
   std::cout <<  " *********************************\n";

}

typedef enum
{
   eModeAerienSift,
   eModeArchiSift,
   eModeBasReliefSift
} eModePorto;

/*********************************************/
/*                                           */
/*            cAppliPorto                    */
/*                                           */
/*********************************************/

class cAppliPorto : public cAppliBatch
{
    public :
       cAppliPorto(int argc,char ** argv);

    private :
       double Dim(const std::string & aName);
       std::string NameParam(int aK);
       void Exec();
       // void Sys(const std::string & aStr,bool Svp=false);
        double  Profondeur(const std::string & aNameFile);



        std::string NameOriInit(const std::string&);
        std::string NameOrient(int aKim,int aKet);
       //  std::string NameHomologues(int aK);

       double Px2(int aKEt);
       double Px1Rel(int aKEt);
       std::string ComMM(int aK);

       double      mPR0;
       double      mPRN;
       double      mPxTrsvInit;
       double      mPxTrsvFinale;
       int         mEt0;
       int         mEt0Spec;
       int         mEtN;
       int         mOneBloc;
       int         mSauvPx;


       double      mSzMin;
       double      mDim;  // Plus petite taille des images
        
	// Facteur dont l'image est trop petite et par lequel il faut 
	// multiplier (ou div) certain param, Par exemple si 2000 est mSzMin et
	// que les images font 400*690 alors mFactMult=5
       double      mFactMult;
       eModePorto  mMode;
};

double  cAppliPorto::Profondeur(const std::string & aNameFile)
{
   return StdGetOC(aNameFile).Externe().Profondeur().Val();
}



/*
std::string cAppliPorto::NameHomologues(int aKet)
{
   return     std::string("MicMac-PL") +ToString(aKet)
	   +  std::string("-") + StdPrefix(mN1)
	   +  std::string("_") + StdPrefix(mN2) 
	   +  std::string(".tif");
}
*/

std::string cAppliPorto::NameParam(int aK)
{
    return  std::string(" applis/XML-Pattron/param-Liaison-")+ToString(aK)+".xml";
/*
   if ((aK != mEt0) || (mEt0==0))
      return  std::string(" applis/XML-Pattron/param-Liaison-")+ToString(aK)+".xml";

   if (mEt0==3)
      return   std::string(" applis/XML-Pattron/param-Liaison-0123.xml");

   ELISE_ASSERT(false,"cAppliPorto::NameParam")
   return "";
   */
}

std::string cAppliPorto::ComMM(int aK)
{
    std::string aRes =
         std::string("bin/MICMAC")
       // + std::string(" applis/XML-Pattron/param-Liaison-"+ToString(aK)+".xml")
       + NameParam(aK)
       + ComCommune()
       + std::string(" @Px1IncCalc=") + ToString(0.0)
       + std::string(" @Px1PropProf=") + ToString(Px1Rel(aK))
       + std::string(" @Px2IncCalc=") + ToString(Px2(aK))
     ;


    if (mMode==eModeBasReliefSift)
    {
        aRes = aRes + " @X_DirPlanInterFaisceau=0"
                    + " @Y_DirPlanInterFaisceau=0"
                    + " @Z_DirPlanInterFaisceau=-1"
                    + " @EstimPxPrefZ2Prof=true"
		    + " @SingulariteInCorresp_I1I2=true";

    }

     if (aK !=0)
     {
        if (aK==mEt0)
	{
	    aRes = aRes + " \%NGI-Ori=true" + " TmpGeom=";
	}
	else
	{
	    aRes = aRes + " \%NGI-Prec-1=true \%NGI-Prec-2=true";
	}
     }

     if (mOneBloc)
        aRes = aRes + std::string(" @SzDalleMin=100000 @SzDalleMax=100000");

    if (aK <=2)
    {
       int aPC = ElMax(1,round_ni(10/mFactMult));

       aRes = aRes + std::string(" @PasCalcul=") + ToString(aPC);
    }

    if ((aK>=2) && ((mMode == eModeArchiSift) || (mMode==eModeBasReliefSift)))   // 
    {
       int aFreqPI = (aK==2) ? 10 : 40;
       std::string aStrFPI = " \%FEPI="+ToString(aFreqPI);
       std::string aStrPAS = " \%PCPI="+ToString(aFreqPI);
       aRes = aRes + " \%SzW=1"  + aStrFPI + aStrPAS;
    }

    if ((aK>=2) && ((mMode == eModeArchiSift)|| (mMode==eModeAerienSift) || (mMode==eModeBasReliefSift)))
    {
       aRes = aRes +  " \%RME=true" ;
    }

    return  aRes;
}



cAppliPorto::cAppliPorto(int argc,char ** argv) :
   cAppliBatch
   (
       argc, argv,
       3, 2,
       "Micmac-LIAISON",
       "PortoHom"
   ),
   mPR0           (0.3),
   mPRN           (0.15),
   mPxTrsvInit    (200),
   mPxTrsvFinale  (10),
   mEt0Spec       (-1),
   mEtN           (3),
   mOneBloc       (0),
   mSauvPx        (0),
   mSzMin         (2000.0)
{
    std::string aNameMode;
    ElInitArgMain
    (
           ARGC(),ARGV(),
           LArgMain() << EAM(aNameMode),
           LArgMain() << EAM(mPR0,"PR0",true)
	              << EAM(mPRN,"PRN",true)
	              << EAM(mPxTrsvInit,"TPx0",true)
	              << EAM(mPxTrsvFinale,"TPxF",true)
		      << EAM(mEt0Spec,"Begin",true)
		      << EAM(mEtN,"End",true)
		      << EAM(mOneBloc,"1Bloc",true)
		      << EAM(mSauvPx,"SauvPx",true)
		      << EAM(mSzMin,"SzMin",true)
    );

    if (aNameMode=="AES")
    {
       mMode = eModeAerienSift;
    }
    else if (aNameMode=="Archi")
    {
       mMode = eModeArchiSift;
    }
    else if (aNameMode=="BasRelief")
    {
       mMode = eModeBasReliefSift;
    }
    else
    {
       ELISE_ASSERT(false,"Unknown mode");
    }

    if (mSauvPx)
    {
       AddPatSauv("Px*Num13*");
       AddPatSauv("Z_Num13_*");
    }
    AddPatSauv("MicMac-PL3-*.tif");
}


std::string cAppliPorto::NameOriInit(const std::string& aName)
{
   return     DirChantier() + ICNM()->Assoc1To1("OrInit",aName,true);
}



std::string cAppliPorto::NameOrient(int aKIm,int aKet)
{
   if (aKet==0)
      return NameOriInit(CurF(aKIm));

   return   DirTmp() 
         +  "OriRelStep" + ToString(aKet-1)  + "_"
	 +  StdPrefix(CurF(aKIm))+ ((aKIm ==0) ? "_For_" : "_On_")
	 +  StdPrefix(CurF(1-aKIm)) + ".xml";
}

double cAppliPorto::Px2(int aKEt)
{
   if (aKEt==0)
      return ElMax(mPxTrsvInit/mFactMult,10.0);
  if (aKEt==1)
     return ElMax(mPxTrsvFinale,mPxTrsvInit/(2.0*mFactMult));
  return mPxTrsvFinale;
}

double cAppliPorto::Px1Rel(int aKEt)
{
    return ((aKEt<=1) ? mPR0 : mPRN) ;
}

double cAppliPorto::Dim(const std::string & aName)
{
   Tiff_Im  aF = Tiff_Im::StdConv(DirChantier()+aName);

   Pt2di aSz = aF.sz();
   return ElMin(aSz.x,aSz.y);
}


void cAppliPorto::Exec()
{

   mDim = ElMin(Dim(CurF1()),Dim(CurF2()));
   mFactMult =  ElMax(1.0,mSzMin/mDim);

   // std::cout << mFactMult << "\n"; getchar();
   if (mEt0Spec==-1)
   {
       if (
                (mDim<400)
	     || (mMode == eModeAerienSift)
	     || (mMode == eModeArchiSift)
	     || (mMode == eModeBasReliefSift)
          )
          mEt0 = 3;
       else
       {
          mEt0 = 0;
       }
   }
   else
   {
      mEt0 = mEt0Spec;
   }


   for (int aKet=mEt0; aKet <= mEtN ; aKet++)
   {
      int aRes = System(ComMM(aKet),true);

      if (aRes !=0)
      {
           std::cout << "RES =" << aRes << "\n";
           if (aRes==26368)
	   {
	       std::cout << "RECOUVREMENT INSUFFISANT \n";
	   }
	   else
	   {
	          std::cout  << "------FAIL IN : \n";
		  std::cout << ComMM(aKet) << "\n";
		  exit(-1);
	   }
      }
   }
}


   //===========================================

int main(int argc,char ** argv)
{
    // system("make -f MakeMICMAC");

    cAppliPorto aAP(argc,argv);

    aAP.DoAll();

    // std::cout << aAP.Com(0) << "\n";
    // std::cout << aAP.Com(1) << "\n";
    // std::cout << aAP.Com(2) << "\n";
    // std::cout << aAP.Com(3) << "\n";

    Banniere_Porto();

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
