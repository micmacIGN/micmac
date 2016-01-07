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
#include "../src/uti_phgrm/MICMAC/MICMAC.h"

/*********************************************************/
/*                                                       */
/*                 InitNadirRank                         */
/*                                                       */
/*********************************************************/


static bool IsPBUG(const Pt2di & aP)
{
    return false;
/*
   return       (aP==Pt2di(97,293))
            ||  (aP==Pt2di(98,293))
         ;
*/
}

void cAppliMICMAC::InitNadirRank()
{
   if (! mMakeMaskImNadir) return;

   std::cout << "cAppliMICMAC::InitNadirRank  " << mGeomDFPxInit << "\n";

   cGeomDiscFPx aGDF = *mGeomDFPxInit;
   aGDF.SetDeZoom(mAnaGeomMNT->AnamDeZoomMasq().Val());
   Pt2di aSz = aGDF.SzDz();

   int aKB = mMakeMaskImNadir->KBest();
   std::string aNameKNad = FullDirMEC()+"AnamImGlobKAngle_K" +ToString(aKB) + ".tif";
   if (! ELISE_fp::exist_file(aNameKNad))
   {
      TIm2D<REAL4,REAL8> aImKNad(aSz);
      Pt2di aPDisc;
      std::vector<double> aVAngles;
      for (aPDisc.x=0 ; aPDisc.x<aSz.x ; aPDisc.x++)
      {
          for (aPDisc.y=0 ; aPDisc.y<aSz.y ; aPDisc.y++)
          {
               aVAngles.clear();
               Pt2dr aPTer = aGDF.DiscToR2(aPDisc);
               for (int aKV=0 ; aKV<int(mPrisesDeVue.size()) ; aKV++)
               {
                   double anA = mPrisesDeVue[aKV]->Geom().IncidTerrain(aPTer);
                   if (anA >=0)
                     aVAngles.push_back(anA);
               }

               // aImKNad.oset(aPDisc,KthVal(&(aVAngles[0]),aVAngles.size(),aKB));
               if (IsPBUG(aPDisc))
               {
                   std::cout << "========  " << aPDisc  << " N: " << aVAngles.size() << " K: " << aKB << " ====\n";
                   for (int aKV=0 ; aKV<int(aVAngles.size()) ; aKV++)
                       std::cout << "  " << aVAngles[aKV] << "\n";
                    
               }
               double aVal = KthValGen(&(aVAngles[0]), (int)aVAngles.size(), aKB, 0.0);
               aImKNad.oset(aPDisc,aVal);
               if (IsPBUG(aPDisc))
               {
                   std::cout <<  "  ##  " << aVal <<  " " << aKB << " ##\n";
               }
          }
      }

      Tiff_Im::CreateFromIm(aImKNad._the_im,aNameKNad);
       std::cout << aGDF.P0() << " " <<  aGDF.P1()  << aGDF.SzDz() << "\n";
   }

   {
      TIm2D<REAL4,REAL8> aImKNad(Pt2di(1,1));
      for (int aKV=0 ; aKV<int(mPrisesDeVue.size()) ; aKV++)
      {
          bool aLoad = false;
          if (! mPrisesDeVue[aKV]->Geom().MasqImNadirIsDone())
          {
              if (! aLoad)
              {
                  aLoad = true;
                  aImKNad = TIm2D<REAL4,REAL8>(Im2D_REAL4::FromFileStd(aNameKNad));
              }
              mPrisesDeVue[aKV]->Geom().DoMasImNadir(aImKNad,aGDF);
          }
      }
   }


}

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

//'class cPurgeAct' does not have a copy constructor 
//which is recommended since the class contains a pointer to allocated memory.
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

// replace all occurences of i_strToReplace by i_strReplacing in i_src
string replace( const std::string &i_src, const std::string &i_strToReplace, const std::string &i_strReplacing )
{
    string res = i_src;
    size_t pos = 0;
    while ( true )
    {
        if ( (pos=res.find(i_strToReplace,pos))==string::npos ) return res;
        res.replace( pos, i_strToReplace.length(), i_strReplacing);
    }
}

// replace strings in the i_stringsToProtect list by their protected version (s -> protect_spaces(s))
//
// assez laid mais je ne vois pas d'autre solution sans specialiser/typer les parties sensibles des commandes (noms de fichiers ici, peut-�tre plus)
// ca a le merite d'etre compatible avec les XML deja existants
void protect_spaces_in_strings( cCmdExePar &io_cmdExePar, const list<string> &i_stringsToProtect )
{
   list<string>::const_iterator itString = i_stringsToProtect.begin();
   std::list<cOneCmdPar> &cmdPar = io_cmdExePar.OneCmdPar();
   list<cOneCmdPar>::iterator itCmdPar;
   list<string>::iterator itCmdSer;
   while ( itString!=i_stringsToProtect.end() )
   {
      string protectedString = protect_spaces( *itString );
      if ( protectedString!=*itString )
      {
	 for ( itCmdPar=cmdPar.begin(); itCmdPar!=cmdPar.end(); itCmdPar++ )
	 {
	    list<string> &cmdSer = itCmdPar->OneCmdSer();
	    for ( itCmdSer=cmdSer.begin(); itCmdSer != cmdSer.end(); itCmdSer++ )
	       *itCmdSer = replace( *itCmdSer, *itString, protectedString );
	 }
      }
      itString++;
   }
}

void print_cmdPar( const cCmdExePar &i_cmdExePar )
{
   const std::list<cOneCmdPar> &cmdPar = i_cmdExePar.OneCmdPar();
   list<cOneCmdPar>::const_iterator itCmdPar;
   list<string>::const_iterator itCmdSer;
   
   for ( itCmdPar=cmdPar.begin(); itCmdPar!=cmdPar.end(); itCmdPar++ )
   {
      const list<string> &cmdSer = itCmdPar->OneCmdSer();
      for ( itCmdSer=cmdSer.begin(); itCmdSer != cmdSer.end(); itCmdSer++ )
	 cout << "[" << *itCmdSer << "]" << endl;
   }
}

void cAppliMICMAC::DoPostProcess()
{
   if (! PostProcess().IsInit()  || (CalledByProcess().Val()))
      return;
// std::cout << "xxx cAppliMICMAC::DoPostProcess  \n"; getchar();

   list<string> strings_to_protect;
   strings_to_protect.push_back( MMDir() );
   strings_to_protect.push_back( NameChantier() );
   protect_spaces_in_strings( PostProcess().Val(), strings_to_protect );
   
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

    double aPas1 = aFPx1.ComputedPas() * aDZ;
    double aPas2 = aFPx2.ComputedPas() * aDZ;

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

    cGeomTerWithMNT2Image aGTWM2I(aResol,*mGeomDFPxInit,PDV2()->Geom(),aDpx);

    Pt2di aSzRes = round_ni(Pt2dr(PDV2()->SzIm())/aResol);
    Im2D_REAL4 aRes(aSzRes.x,aSzRes.y);
    Pt2di aPIm;
    for (aPIm.x=0 ; aPIm.x<aSzRes.x ; aPIm.x++)
    {
        for (aPIm.y=0 ; aPIm.y<aSzRes.y ; aPIm.y++)
        {
	   Pt2dr aPax = aGTWM2I.Im2Px(Pt2dr(aPIm));
	   aRes.data()[aPIm.y][aPIm.x] = (REAL4)aPax.y;
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

const cMMUseMasq3D * cAppliMICMAC::Masq3DOfEtape(cEtapeMecComp & anEtape)
{
   if (! MMUseMasq3D().IsInit()) return 0;
   const cMMUseMasq3D & aMasq =  MMUseMasq3D().Val();

   if (anEtape.EtapeMEC().DeZoom() > aMasq.ZoomBegin().Val()) return 0;

   return & aMasq;
}

void cAppliMICMAC::DoMasq3D(cEtapeMecComp & anEtape,const cMMUseMasq3D & aMasq)
{
   // const cEtapeMEC &   anEM = anEtape.EtapeMEC();
   Tiff_Im  aFM = anEtape.FileMask3D2D();
   std::string aNameNuage =     aMasq.PrefixNuage().IsInit() ?
                                (mFullDirMEC+ aMasq.PrefixNuage().Val() + "_Etape_" + ToString(anEtape.Num()) + ".xml") :
                                anEtape.NameXMLNuage()  ;
   std::string aCom = MM3dBinFile("TestLib") 
                     + " Masq3Dto2D " 
                     + aMasq.NameMasq() + std::string(" ")
                     + aNameNuage  + std::string(" ")
                     + aFM.name() ;

/*
std::cout << "aFM.name"  << aFM.name() << "##" << anEtape.NameXMLNuage()  << "\n";
*/
//  std::cout << aCom << "\n";
//  getchar();

   System(aCom);
// getchar();




   // mNameXMLNuage
   // std::string aCom = 

}

void cAppliMICMAC::MakeResultOfEtape(cEtapeMecComp & anEtape)
{

 
   MakeGenCorPxTransv(anEtape);
   std::list<string> mVProcess;
   const cEtapeMEC &   anEM = anEtape.EtapeMEC();
   
   const cMMUseMasq3D * aMasq3D = Masq3DOfEtape(anEtape);
   if (aMasq3D)
   {
       DoMasq3D(anEtape,*aMasq3D);
   }



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

    if (DoMTDNuage())
    {
       anEtape.DoRemplitXMLNuage();
    }
    MakeDequantSpecial();
}


void cAppliMICMAC::MakeDequantSpecial()
{
   // MPD Modif , le dequant tel que implemante dans MicMac a pas mal d'effet de bord,
   // pour les reduire efficacement, il faudraoit augmenter taille des recouvrt et 
   // taille image mais ensite MicMac est couteux en memoire. Donc :
   //    * on relance un Dequant "basique"
   //    * on laisse ce qui est fait actuellement car il y a pas mal d'autre chose pour preparer
   //      ortho etc .... donc on prend pas de risques ...

   if (!mCurEtape->IsOptDequant())
       return;

    if (DoNothingBut().IsInit())
    {
        return;
    }

   if (! mDoTheMEC)
       return;

   if (mPrecEtape==0) return;
   if (mPrecEtape->IsOptDequant()) return;
     
   for (int aKPx=0 ; aKPx<DimPx() ; aKPx++)
   {
       std::string aTifQuant  =    mPrecEtape->KPx(aKPx).NameFile();
       std::string aTifDeqQuant  = mCurEtape->KPx(aKPx).NameFile();

       double aR=0;
       double aOfs = 0;


       // std::cout << "AAAAAAAAAAAAAa IsOptDequant " << mPrecEtape << " " << mCurEtape << "\n";
       if (DimPx()==1)
       {
           // cFileOriMnt aFOMQ   =  StdGetFromPCP(StdPrefix(aTifQuant)+".xml"   ,FileOriMnt);
           // cFileOriMnt aFOMDeQ =  StdGetFromPCP(StdPrefix(aTifDeqQuant)+".xml",FileOriMnt);

           cFileOriMnt aFOMQ   =  StdGetFromPCP(NameOrientationMnt(mPrecEtape)  ,FileOriMnt);
           cFileOriMnt aFOMDeQ =  StdGetFromPCP(NameOrientationMnt(mCurEtape),FileOriMnt);
       // Z = a0 + R0 z0
       // Z = a1 + R1 z1
       //  z1 = (Z-a1) / R1 = (a0 +R0 z0 -a1) /R1 = (a0-a1) /R1  + z1 (R0/R1)


//    Z_Num7_DeZoom2_STD-MALT.xml
//      <ResolutionAlti>0.0135</ResolutionAlti>
//    Z_Num8_DeZoom2_STD-MALT.xml
//       <ResolutionAlti>0.003</ResolutionAlti>


           aR = aFOMQ.ResolutionAlti() / aFOMDeQ.ResolutionAlti();
           aOfs = (aFOMQ.OrigineAlti()-aFOMDeQ.OrigineAlti()) /aFOMDeQ.ResolutionAlti();
           // std::cout << " BB  " << aTifQuant  << " " << aFOMQ.ResolutionAlti() << "\n";
           // std::cout << " CCC " << aTifDeqQuant << " " << aFOMDeQ.ResolutionAlti() << "\n";
           // std::cout << " DDDD,  R=" << aR << " Of=" << aOfs << "\n";
       }
       else
       {
              double aPasQ =  mPrecEtape->KPx(aKPx).ComputedPas();
              double aPasDeQ =   mCurEtape->KPx(aKPx).ComputedPas();
              aR=  aPasQ/aPasDeQ;
              aOfs=0;

             // std::cout << " EEEEE,  R=" << aR << " Of=" << aOfs << "\n";
       }

       std::string aCom =   MM3dBinFile("Dequant")
                          + " "  +  aTifQuant
                          + " Out=" + aTifDeqQuant
                          + " Dyn=" + ToString(aR)
                          + " Offs=" + ToString(aOfs)
                          + " SzRecDalles=1500"
                          + " SzMaxDalles=15000" ;
       // std::cout << "COM= " << aCom << "\n";
       System(aCom);
   }
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
   aMod.Pas().x = anEtape.KPx(0).ComputedPas();
   aMod.Pas().y = anEtape.KPx(1).ComputedPas();

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

   aNameBin = string("\"")+aNameBin + "\" "  +  aNameTmp;
   aNameBin = aNameBin + " Ch1="+ ChMpDCraw(PDV1())+" Ch2="+ ChMpDCraw(PDV2());
   ::System( aNameBin);
   
   ELISE_fp::RmFile( aNameTmp );
}





/*Footer-MicMac-eLiSe-25/06/2007

Ce logiciel est un programme informatique servant �  la mise en
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
associés au chargement,  �  l'utilisation,  �  la modification et/ou au
développement et �  la reproduction du logiciel par l'utilisateur étant 
donné sa spécificité de logiciel libre, qui peut le rendre complexe �  
manipuler et qui le réserve donc �  des développeurs et des professionnels
avertis possédant  des  connaissances  informatiques approfondies.  Les
utilisateurs sont donc invités �  charger  et  tester  l'adéquation  du
logiciel �  leurs besoins dans des conditions permettant d'assurer la
sécurité de leurs systèmes et ou de leurs données et, plus généralement, 
�  l'utiliser et l'exploiter dans les mêmes conditions de sécurité. 

Le fait que vous puissiez accéder �  cet en-tête signifie que vous avez 
pris connaissance de la licence CeCILL-B, et que vous en avez accepté les
termes.
Footer-MicMac-eLiSe-25/06/2007*/
