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
#include "MICMAC.h"
#include "im_tpl/image.h"

namespace NS_ParamMICMAC
{

/*********************************************************/
/*                                                       */
/*                 cDblePx                               */
/*                                                       */
/*********************************************************/



cDblePx::cDblePx(Pt2di aSz) :
   mPx1    (aSz.x,aSz.y),
   mTPx1   (mPx1),
   mPx2    (aSz.x,aSz.y),
   mTPx2   (mPx2)
{
}

/*********************************************************/
/*                                                       */
/*         cGeomTerWithMNT2Image                         */
/*                                                       */
/*********************************************************/

Pt2dr cGeomTerWithMNT2Image::Direct(Pt2dr aPTer) const
{
   double aVPx[2];
   aVPx[0] = mPx.mTPx1.getprojR(aPTer);
   aVPx[1] = mPx.mTPx2.getprojR(aPTer);
   aPTer = aPTer*mResol;
   aPTer = mGeomTer.RDiscToR2(aPTer);
   return  mGeomI.Objet2ImageInit_Euclid(aPTer,aVPx)/ mResol;
}

cGeomTerWithMNT2Image::cGeomTerWithMNT2Image
(
     const double &             aResol,
     const cGeomDiscFPx &       aGeomTer,
     const cGeomImage &         aGeomI,
     const cDblePx    &         aDblPx
)  :
     mResol   (aResol),
     mGeomTer (aGeomTer),
     mGeomI   (aGeomI),
     mPx      (aDblPx)
{
}

Pt2dr cGeomTerWithMNT2Image::ImAndPx2Ter(Pt2dr aPIm,Pt2dr aPax)
{
   double aVPx[2];
   aVPx[0] = aPax.x;
   aVPx[1] = aPax.y;

   aPIm = aPIm * mResol;
   Pt2dr aPter = mGeomI.ImageAndPx2Obj_Euclid(aPIm,aVPx);
   return aPter/mResol;
}

Pt2dr cGeomTerWithMNT2Image::ImAndPx2Px(Pt2dr aPIm,Pt2dr aPax)
{
   Pt2dr aPTer = ImAndPx2Ter(aPIm,aPax);

   return Pt2dr(mPx.mTPx1.getprojR(aPTer),mPx.mTPx2.getprojR(aPTer));
}

Pt2dr cGeomTerWithMNT2Image::Im2Px(Pt2dr aPIm)
{
   const double * aVPx = mGeomTer.V0Px();
   Pt2dr aPax(aVPx[0],aVPx[1]);

   for (int aK=0 ; aK<3 ; aK++)
      aPax = ImAndPx2Px(aPIm,aPax);

   if (0) // Verif
   {
        Pt2dr aPTer = ImAndPx2Ter(aPIm,aPax);
	Pt2dr aPIm2 = Direct(aPTer);
	std::cout << "D(" << aPIm << ")=" << euclid(aPIm,aPIm2) << "\n";
   }

   return aPax;
} 

void  cGeomTerWithMNT2Image::Diff(ElMatrix<REAL> &,Pt2dr) const
{
   ELISE_ASSERT(false,"cGeomTerWithMNT2Image::Diff");
}


// Pt2dr cGeomTerWithMNT2Image::ImPx2Px(Pt2dr aPIm,Pt2dr aPax)

/*********************************************************/
/*                                                       */
/*                cPurgeAct,....                         */
/*                                                       */
/*********************************************************/

class cPurgeAct
{
    public :
       cPurgeAct(const std::string & aSelec,bool toSuppres) :
          mAutom  (new cElRegex(aSelec,2)),
          mToSupr (toSuppres)
       {
          ELISE_ASSERT(mAutom->IsOk(),"Cannot Compile cPurgeAct::Autom");
       }
       ~cPurgeAct() 
       {
           delete mAutom;
       }
       void ModifDecision(const std::string & aName,bool & aDecPrec) const
       {
          if (mAutom->Match(aName))
          {
             aDecPrec = mToSupr;
          }
       }
    private :
       cElRegex * mAutom;
       bool       mToSupr;
};

class cParseDir_PurgeFile : public ElActionParseDir
{
     public :
         cParseDir_PurgeFile(const std::list<cPurgeAct *> & aLPA) :
              mLPA (aLPA)
         {
         }
         void act(const ElResParseDir & aRPD)
         {
             bool isSel = false;
             std::string aName = aRPD.name();
             for 
             (
                 std::list<cPurgeAct *>::const_iterator itPA = mLPA.begin();
                 itPA !=  mLPA.end();
                 itPA++
             )
             {
                (*itPA)->ModifDecision(aName,isSel);
             }
             if (isSel) ELISE_fp::RmFile( aName );
         }
     private :
         const std::list<cPurgeAct *> & mLPA;
};


/*********************************************************/
/*                                                       */
/*                cAppliMICMAC                           */
/*                                                       */
/*********************************************************/

void cAppliMICMAC::DoPostProcess()
{
   if (! PostProcess().IsInit()  || (CalledByProcess().Val()))
      return;
// std::cout << "xxx cAppliMICMAC::DoPostProcess  \n"; getchar();
   DoCmdExePar(PostProcess().Val(),ByProcess().Val());
}

void cAppliMICMAC::DoPurgeFile()
{
   if ( (! ActivePurge().Val()) || (CalledByProcess().Val()))
      return;

   std::list<cPurgeAct *> mLPA;
   for
   (
       std::list<cPurgeFiles>::const_iterator itPF = PurgeFiles().begin();
       itPF != PurgeFiles().end();
       itPF++
   )
   {
       mLPA.push_back
       (
            new cPurgeAct
                (
                      itPF->PatternSelPurge(),
                      itPF->PurgeToSupress()
                )
       );
   }

   cParseDir_PurgeFile aPD_PF(mLPA);
   ElParseDir(WorkDir().c_str(),aPD_PF);
       


   DeleteAndClear(mLPA);
}

/*
    Cette section rassemble des resultats qui sont genérés potentiellement
  après chaque étape de mise en correspondance.

*/



void cAppliMICMAC::MakeImagePx8Bits
     (
          std::list<std::string> &    mVProcess,
          const cEtapeMecComp &         anEtape,
          int                           aKFile,
          const cTplValGesInit<bool> &  Genere,
          const cTplValGesInit<int>  &  anOffset,
          const cTplValGesInit<double>& aDyn
     )
{
   if (DoNothingBut().IsInit())
      return;

   if (! Genere.ValWithDef(false))
      return;

   ELISE_ASSERT
   (
      (aKFile < DimPx()),
      "Dim Px in cAppliMICMAC::MakeImagePx8Bits"
   );

/*
   if (! ELISE_fp::exist_file("bin/to8Bits"))
   {
       VoidSystem("make bin/to8Bits");
   }
*/
   std::string aStrOff = "";
   if (anOffset.IsInit())
      aStrOff = std::string(" Offset=") + ToString(anOffset.Val());
      
   std::string aStr =   
           MMBin() + std::string("to8Bits ")
         + anEtape.KPx(aKFile).NameFile()
         + std::string(" VisuAff=0")
         + aStrOff
         + std::string(" Dyn=")    + ToString(aDyn.ValWithDef(1.0));

    for 
    (
          std::list<std::string>::const_iterator itS=anEtape.EtapeMEC().ArgGen8Bits().begin();
          itS!=anEtape.EtapeMEC().ArgGen8Bits().end();
          itS++
    )
    {
       aStr = aStr + " " +*itS;
    }
   
   mVProcess.push_back(aStr);
}


cDblePx cAppliMICMAC::LoadPx(cEtapeMecComp & anEtape,double aResol)
{
    ELISE_ASSERT(mDimPx == 2,"Dim 1 in MakeGenCorPxTransv");

    int aDZ = anEtape.EtapeMEC().DeZoom();
    aResol /= aDZ;

    const  cFilePx & aFPx1 = anEtape.KPx(0);
    const  cFilePx & aFPx2 = anEtape.KPx(1);

    double aPas1 = aFPx1.Pas() * aDZ;
    double aPas2 = aFPx2.Pas() * aDZ;

    Tiff_Im anIm1 = aFPx1.FileIm();
    Tiff_Im anIm2 = aFPx2.FileIm();
    Tiff_Im aMasq = FileMasqOfResol(aDZ);


    Pt2di aSzResTer = round_ni(Pt2dr(anIm1.sz()) / aResol);
    cDblePx aRes(aSzResTer);

    Pt2dr aPResol(aResol,aResol);

    Fonc_Num aFPx = Virgule(aPas1*anIm1.in_proj(),aPas2*anIm2.in_proj());

    Pt2dr aP0(0,0);

    ELISE_COPY
    (
        aRes.mPx1.all_pts(),
	  StdFoncChScale(aFPx*aMasq.in(0),aP0,aPResol)
	/ Max(0.1,StdFoncChScale(aMasq.in(0),aP0,aPResol)),
	Virgule(aRes.mPx1.out(),aRes.mPx2.out())
    );

    return aRes;
}


void cAppliMICMAC::MakeGenCorPxTransv(cEtapeMecComp & anEtape)
{
    if (! anEtape.EtapeMEC().GenCorPxTransv().IsInit())
       return;

    if (DoNothingBut().IsInit()  && (! ButDoGenCorPxTransv().Val()))
       return;

    ELISE_ASSERT(mDimPx == 2,"Dim 1 in MakeGenCorPxTransv");

    Pt2dr aDirT;
    bool aGotT = PDV2()->Geom().DirEpipTransv(aDirT);
    ELISE_ASSERT(aGotT,"Cann't Get Di-Transv  in MakeGenCorPxTransv");


    cGenCorPxTransv aGCPT = anEtape.EtapeMEC().GenCorPxTransv().Val();
    double aResol =  aGCPT.SsResolPx() ;

//std::cout << "ENTER cGeomTerWithMNT2Image\n";
    cDblePx aDpx = LoadPx(anEtape,aResol);
//std::cout << "ENTER cGeomTerWithMNT2Image\n";

    cGeomTerWithMNT2Image aGTWM2I(aResol,mGeomDFPxInit,PDV2()->Geom(),aDpx);

    Pt2di aSzRes = round_ni(Pt2dr(PDV2()->SzIm())/aResol);
    Im2D_REAL4 aRes(aSzRes.x,aSzRes.y);
    Pt2di aPIm;
    for (aPIm.x=0 ; aPIm.x<aSzRes.x ; aPIm.x++)
    {
        for (aPIm.y=0 ; aPIm.y<aSzRes.y ; aPIm.y++)
        {
	   Pt2dr aPax = aGTWM2I.Im2Px(Pt2dr(aPIm));
	   aRes.data()[aPIm.y][aPIm.x] = aPax.y;
        }
	//std::cout << aPIm.x << "\n"; getchar();
    }


    std::string aNameXML = aGCPT.NameXMLFile();


    // Tiff_Im::CreateFromIm(aRes,"../TMP/Px2.tif");
    cCorrectionPxTransverse aCTP;
    aCTP.DirPx() = aDirT;
    aCTP.ValeurPx() = aRes;
    aCTP.SsResol() = aResol;

    MakeFileXML(aCTP,FullDirResult()+aNameXML);
}

void cAppliMICMAC::MakeResultOfEtape(cEtapeMecComp & anEtape)
{
 
   MakeGenCorPxTransv(anEtape);
   std::list<string> mVProcess;
   const cEtapeMEC &   anEM = anEtape.EtapeMEC();
   
   MakeImagePx8Bits
   (
       mVProcess,anEtape,0,
       anEM.Gen8Bits_Px1(),anEM.Offset8Bits_Px1(),anEM.Dyn8Bits_Px1()
   );

   MakeImagePx8Bits
   (
       mVProcess,anEtape,1,
       anEM.Gen8Bits_Px2(),anEM.Offset8Bits_Px2(),anEM.Dyn8Bits_Px2()
   );


   ExeProcessParallelisable(false,mVProcess);

   anEtape.ExportModelesAnalytiques();

   for 
   (
      std::list<cGenereModeleRaster2Analytique>::const_iterator 
           itM=anEtape.EtapeMEC().ExportAsModeleDist().begin();
      itM!=anEtape.EtapeMEC().ExportAsModeleDist().end();
      itM++
   )
   {
      MakeExportAsModeleDist(*itM,anEtape);
   }

    anEtape.DoRemplitXMLNuage();
}


std::string cAppliMICMAC::ChMpDCraw(const cPriseDeVue * aPDV) const
{
    return ICNM()->Assoc2To1("ImChan2SplitMpDCraw",aPDV->Name(),false).second;
}

void cAppliMICMAC::MakeExportAsModeleDist
     (
          const cGenereModeleRaster2Analytique aModeleInit,
	  cEtapeMecComp & anEtape
     )
{
   cGenereModeleRaster2Analytique aMod = aModeleInit;

   aMod.Dir() = FullDirMEC();
   aMod.Im1() = anEtape.KPx(0).NameFileSsDir();
   aMod.Im2() = anEtape.KPx(1).NameFileSsDir();
   aMod.Pas().x = anEtape.KPx(0).Pas();
   aMod.Pas().y = anEtape.KPx(1).Pas();

// std::cout << aMod.Pas().x << " " << aMod.Pas().y << "\n";
// std::cout << "VERIF INVERSION DANS SauvegardeMR2A !! \n";
   ELISE_ASSERT
   (
           aMod.SauvegardeMR2A().IsInit()
        && aMod.SauvImgMR2A().IsInit(),
       " Init of SauvegardeMR2A"
   );

   aMod.NameSauvMR2A() = FullDirResult() + aModeleInit.NameSauvMR2A()
                         +mNameChantier + ".xml";

   aMod.SauvImgMR2A().SetVal(FullDirResult() + aModeleInit.SauvImgMR2A().Val());



   const std::string aNameTmp = "RtYjuiklpM76e4.xml";
   std::string aNameBin = MMDir() + "bin"+ELISE_CAR_DIR+"ModeleRadial";
   MakeFileXML(aMod,aNameTmp);
   RequireBin(mNameExe,aNameBin);

   aNameBin = aNameBin + " "  +  aNameTmp;
   aNameBin = aNameBin + " Ch1="+ ChMpDCraw(PDV1())+" Ch2="+ ChMpDCraw(PDV2());
   ::System( aNameBin);
   
   ELISE_fp::RmFile( aNameTmp );
}




};

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
