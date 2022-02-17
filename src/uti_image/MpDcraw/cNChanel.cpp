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
// #include "anag_all.h"

#include "StdAfx.h"

namespace NS_MpDcraw
{

cNChannel::cNChannel
(
     const cArgMpDCRaw & anArg,
     const std::string &  aFullNameFile,
     Im2D_REAL4 aFullImage,
     int        aNbC,
     const char **    Names,
     Pt2di*     mVP0,
     Pt2di      aPer
)  :
   mArg           (anArg),
   mFullNameFile  (aFullNameFile),
   mCamCorDist    (0),
   mHomograRedr     (0),
   mInvHomograRedr     (0)
{
   SplitDirAndFile(mNameDir,mNameFile,mFullNameFile);

   if (mArg.CamDist() !="")
   {
       mCamCorDist = Std_Cal_From_File(mNameDir+mArg.CamDist());
   }

   if (mArg.HomolRedr() !="")
   {
      ElPackHomologue aPack = ElPackHomologue::FromFile(mNameDir+mArg.HomolRedr());
      if (mCamCorDist)
      {
          for 
          (
             ElPackHomologue::iterator itP =aPack.begin();
             itP!=aPack.end();
             itP++
          )
          {
              itP->P1() = mCamCorDist->DistInverse(itP->P1());
          }
      }
      mHomograRedr = new cElHomographie(aPack,true);
      mInvHomograRedr = new cElHomographie(mHomograRedr->Inverse());
   }


   for (int aKC=0; aKC<aNbC ; aKC++)
      mVCh.push_back(cOneChanel(this,Names[aKC],aFullImage,mVP0[aKC],aPer));
}


//static const char * TheNameC[4] = {"R","V1","V2","B"};
//static const char * TheNameC[4] = {NR,NV1,NV2,NB};




// #define NR "R"
// #define NV1 "V1"
// #define NV2 "V2"
// #define NB "B"


Fonc_Num FilterMax(const cArgMpDCRaw & anArg,Fonc_Num aF,Tiff_Im aFile)
{
   return aF;
/*
   double aFact = 0.7;
   return canny_exp_filt(aF,aFact,aFact) / canny_exp_filt(aFile.inside(),aFact,aFact);
*/
}

Fonc_Num GamCor(const cArgMpDCRaw & anArg,Fonc_Num aF,const std::string & aNameFile)
{
   double aGamma = anArg.Gamma(aNameFile);
   if (aGamma==1.0) return aF;
   if (aGamma>0) return 255.0 * pow(Max(0,aF)/255.0,1/aGamma);

    // double aEps = -aGamma * anArg.EpsLog();
    double aEps = anArg.EpsLog();
    return -aGamma * (log2(aF+aEps)-log2(aEps));
}

void  cNChannel::Split(const cArgMpDCRaw & anArg,const std::string & aPost,Tiff_Im aFileIn)
{
   if (! anArg.DoSplit()) return;

   for (int aK=0 ; aK<2 ; aK++)
   {
       std::string aName = NameRes(aPost,(aK==0) ? "Gauche" : "Droite");

       Pt2di aSzG = aFileIn.sz();
       Pt2di aMExt = anArg.SpliMargeExt();

       Pt2di aSzCl = Pt2di
                     (
                         aSzG.x/2 -    aMExt.x -  anArg.SpliMargeInt(),
                         aSzG.y  - 2 * aMExt.y
                     );

       Pt2di aP0 = aMExt;
       Pt2di aP1 = aP0 + aSzCl;
       if (aK==1)
       {
           aP0.x = aSzG.x - aP0.x;
           aP1.x = aSzG.x - aP1.x;
       }

       Box2di aBox (aP0,aP1);
                     
       Tiff_Im aFileOut
               (
                   aName.c_str(),
                   aBox.sz(),
                   aFileIn.type_el(),
                   Tiff_Im::No_Compr,
                   aFileIn.phot_interp(),
                   anArg.ArgMTD()
                );
       ELISE_COPY
       (
           rectangle(Pt2di(0,0),aBox.sz()),
           trans(aFileIn.in(),aBox._p0),
           aFileOut.out()
       );
   }

}
/*

*/


#define BUGNEF true

// "/home/mpd/MMM/culture3d/bin/mm3d" MpDcraw "./IMGP7501.JPG"  Add16B8B=0  ConsCol=0  ExtensionAbs=None  16B=0  CB=1  NameOut=./Tmp-MM-Dir/IMGP7501.JPG_Ch3.tif UseFF=0 Gamma=1.0 EpsLog=1.0


Fonc_Num  cArgMpDCRaw::FlatField(const cMetaDataPhoto & aMDP,const std::string & aNameFile)
{
   double aZoomFF = 10.0;
   		char foc[15],dia[14];
		sprintf(foc, "%04d", int(round_ni(aMDP.FocMm(true))));
                // MPD : il y a du y avoir une regression car tel quel cela ne peut pas marcher
		// sprintf(dia, "%03d", int(10*round_ni(aMDP.Diaph(true))));
		sprintf(dia, "%03d", int(round_ni(10*aMDP.Diaph(true))));
		std::string aNameFF="Foc" + (string)foc + "Diaph" + (string)dia + "-FlatField.tif";
   //std::string aNameFF = DirOfFile(aNameFile)+ "Foc"+ ToString(round_ni(aMDP.FocMm(true))) + "Diaph" + ToString(10*round_ni(aMDP.Diaph(true))) + "-FlatField.tif";
  
// MPD : si ca ne marche pas avec le flad field specif, on tente un Flat Field basic global
   if (!ELISE_fp::exist_file(aNameFF))
   {
      aNameFF = "FlatField.tif";
   }


   if ((!ELISE_fp::exist_file(aNameFF)) || (! UseFF()))
   {
// Foc0018Diaph060-FlatField.tif
// Foc0018Diaph063-FlatField.tif
// std::cout <<  "  NOFFFFFFf " << aNameFF  << " " << ELISE_fp::exist_file(aNameFF) << " " << UseFF() << "\n";
      return 1;
   }
 std::cout <<  " -----Image corected with flatfield : " << aNameFF << " \n";


   Im2D_REAL4 aFlF=Im2D_REAL4::FromFileStd(aNameFF);

   Pt2dr aDec = (Pt2dr(aMDP.XifSzIm()) - Pt2dr(aFlF.sz()) * aZoomFF) /2.0;

   return   aFlF.in_proj()[Virgule(FX-aDec.x,FY-aDec.y)/aZoomFF];
}



cNChannel cNChannel::Std(const cArgMpDCRaw & anArg,const std::string & aNameFile)
{

   cSpecifFormatRaw * aSFR = GetSFRFromString(aNameFile);
   if (aSFR && !(aSFR->BayPat().IsInit())) aSFR = 0;


   cMetaDataPhoto aMDP = cMetaDataPhoto::CreateExiv2(aNameFile);
   cCameraEntry * anEntry = CamOfName(aMDP.Cam(true));
   bool RBswap = anArg.SwapRB(false);


   bool HasFlF=false;
   Im2D_REAL4 aFlF(1,1);
   char foc[50],dia[40]; // !!!! SINON DEBORDEMENT 
   sprintf(foc, "%04d", int(round_ni(10*aMDP.FocMm(true))));
   sprintf(dia, "%03d", int(round_ni(10*aMDP.Diaph(true))));
   std::string aNameFF="Foc" + (string)foc + "Diaph" + (string)dia + "-FlatField.tif";
   //std::string aNameFF = DirOfFile(aNameFile)+ "Foc"+ ToString(round_ni(aMDP.FocMm())) + "Diaph" + ToString(round_ni(10*aMDP.Diaph(true))) + "-FlatField.tif";
   // Pas de FF en coul pour l'insntnt
   if (ELISE_fp::exist_file(aNameFF) &&  (!  anArg.Cons16B()) && anArg.UseFF() )

   {
       std::cout << "USE FLAT FIELD " << aNameFF << "\n";
       aFlF=Im2D_REAL4::FromFileStd(aNameFF);
       HasFlF=true;
   }
   else 
   {
       std::cout << "NO FLAT FIELD " << aNameFF << "\n";
   }

    const char * NR = "R";
    const char * NV1 = "V";
    const char * NV2 = "W";
    const char * NB = "B";
    const char * TheNameC[4] = {"R","V","W","B"};

    {
        std::string aBayPat = aMDP.BayPat();
        const char * CBP = aBayPat.c_str();
        for (int aK= 0 ; aK< 4 ; aK++)
        {
            std::string * aNewStr = new std::string;
            *aNewStr += CBP[aK];
            TheNameC[aK] = aNewStr->c_str();
        }
    }




    bool M16B =      anArg.Cons16B() 
                  || anArg.Adapt8B() 
                  || (anArg.Dyn()!=0) 
                  || (anArg.ExpTimeRef()!=0) 
                  || (anArg.DiaphRef()!=0) 
                  || (anArg.IsoSpeedRef()!=0) 
                  // || HasFlF  MPD Avril 2013 : comprend pas, ce fait saturer les images !!
                ;

   if (aSFR) M16B = (aSFR->NbBitsParPixel()>=16);

    // cNChannel  aNC(anArg,aNameFile,anIm,4,TheNameC,&(aVP0[0]),Pt2di(2,2));

    std::string aNameTmp = StdPrefix(aNameFile)+ "XXX-hkjyur-toto.pgm";
    // std::string aNameCom =  std::string("dcraw -t 0 -c -d  " )
    // ElDcraw -d -c -t 0 SiteU-Drone-DSC07977.ARW > T.pgm

    bool  TraitBasic =  (anEntry && anEntry->DevRawBasic().Val()) ;

    std::string Options = " -d  "  ;

    if (TraitBasic)
       Options = "" ;

    if (aSFR)
    {
       aNameTmp = aNameFile;
    }
    else
    {
         std::string aNameCom = MM3dBinFile_quotes("ElDcraw")
                           + std::string("-c -t 0 ") + Options
                           + std::string(M16B?" -4 ":"")
                           + ToStrBlkCorr(aNameFile) + " > " + aNameTmp;
          System(aNameCom.c_str());
     }
     

     std::vector<Im2DGen *>  aV3;
     Im2D_REAL4              anIm(1,1);
     Pt2di aSz(1,1);


			   

    if (aSFR)
    {
        Tiff_Im aTifTmp = Elise_Tiled_File_Im_2D::XML(aNameFile).to_tiff();
        aSz = aTifTmp.sz();
        anIm.Resize(aSz);
        ELISE_COPY(anIm.all_pts(),aTifTmp.in(),anIm.out());
        aV3.push_back(&anIm);
    }
    else
    {
       if (TraitBasic)
       {
          Tiff_Im aTifTmp = Tiff_Im::StdConv(aNameTmp);
          aV3 = aTifTmp.ReadVecOfIm();
          aSz = aV3[0]->sz();
       }
       else
       {
           anIm = Im2D_REAL4::FromFileBasic(aNameTmp);
           aSz = anIm.sz();
           aV3.push_back(&anIm);
       }


        // DCraw genere les ppm en MSBF 
        if ( M16B && (!MSBF_PROCESSOR()))
        {
            for (int aK=0 ; aK<int(aV3.size()) ; aK++)
            {
               Im2DGen & anIm = *(aV3[aK]);
               Fonc_Num f = Iconv(anIm.in());
               ELISE_COPY
	       (
	           anIm.all_pts(),
                   256 * (f&255) + f/256,
	           anIm.out()
	       );
            }
        }
    }

    std::vector<Pt2di>  aVP0;
    aVP0.push_back(Pt2di(0,0));
    aVP0.push_back(Pt2di(1,0));
    aVP0.push_back(Pt2di(0,1));
    aVP0.push_back(Pt2di(1,1));

    cNChannel  aNC(anArg,aNameFile,anIm,4,TheNameC,&(aVP0[0]),Pt2di(2,2));

    Im2D_REAL8 aPds( 3,3,
	             " 1 2 1" 
                     " 2 4 2" 
                     " 1 2 1"
                   );
    double aSomPds = 16.0;
    ELISE_COPY(aPds.all_pts(),aPds.in()/aSomPds,aPds.out());


    std::vector<double> aOfs;
    double aOfsPar = anArg.Offset();
    // if (! m16B) aOfsPar /= 64.0;  // Assez empirique
    for (int aK=0 ; aK<3 ; aK++)
    {
        aOfs.push_back(aOfsPar);
    }
    double aSof = (aOfs[0] + 2* aOfs[1] + aOfs[2])/4.0;

    std::vector<double> aWB = anArg.WB();
    double aFact = (anArg.Dyn()!=0) ? anArg.Dyn() : 1.0;

// std::cout << "NFFFF " << aNameFile << "\n";
// std::cout << "NFFFF " << aNameFile << "\n";
    if (anArg.ExpTimeRef() != 0)
    {
 // std::cout << "Ttttt " << anArg.ExpTimeRef() << " " << aMDP.ExpTime() << "\n";
       aFact = aFact  * (anArg.ExpTimeRef()/aMDP.ExpTime());
    }

    if (anArg.DiaphRef() != 0)
    {
       aFact = aFact  * ElSquare(aMDP.Diaph()/anArg.DiaphRef());
    }

    if (anArg.IsoSpeedRef() != 0)
    {
       aFact = aFact  * (anArg.IsoSpeedRef()/aMDP.IsoSpeed());
    }

   std::cout << "FACT COrr = " << aFact << "\n";


    if (  anArg.BayerCalib() && anArg.BayerCalib()->WB().IsInit() && (! anArg.WBSpec()))
    {
       Pt3dr aWB2 = anArg.BayerCalib()->WB().Val();
       aWB[0] = aWB2.x ;
       aWB[1] = aWB2.y ;
       aWB[2] = aWB2.z ;
    }
    for (int aK=0 ; aK<3 ; aK++)
        aWB[aK] *= aFact;


    if (anArg.GrayBasic() || (TraitBasic && anArg.GrayReech()))
    {
        std::string aName =  aNC.NameRes("GB");

        Tiff_Im aFile
                (
                   aName.c_str(),
                   aSz,
                   aNC.TypeOut(false,aSFR),
                   Tiff_Im::No_Compr,
                   Tiff_Im::BlackIsZero,
                   anArg.ArgMTD()
                );
         Im2D_REAL4  aITMP(aSz.x,aSz.y);
         double aVMax;
         Fonc_Num aF=0;
         if (TraitBasic)
         {
              aF =  (aV3[0]->in_proj() + aV3[1]->in_proj() +aV3[2]->in_proj()) / 3.0;
         }
         else
         {
             aF = som_masq(anIm.in_proj(),aPds);
         }
         aF = aF -aSof;
         aF = aF *  aFact;

         if (HasFlF)
         {
              aF = aF / aFlF.in_proj()[Virgule(FX,FY)/10.0];
         }
         aF = GamCor(anArg,aF,aNameFile); 


         ELISE_COPY
	 (
	     aV3[0]->all_pts(),
	     Virgule(aF,FilterMax(anArg,aF,aFile)),
	     Virgule(aITMP.out() , VMax(aVMax))
	 );

         aF = aITMP.in();
         if ( anArg.Adapt8B())
            aF = aITMP.in() * (255.0/aVMax) ;

	 aF = Tronque(aNC.TypeOut(false,aSFR),aF);
         ELISE_COPY(aV3[0]->all_pts(),aF,aFile.out());

         aNC.Split(anArg,"GB",aFile);
    }  

      // if (anArg.GrayBasic() || (TraitBasic && anArg.GrayReech()))
      // if (anArg.ColReech())
    if (anArg.ColBasic() || (TraitBasic && anArg.ColReech()))
    {
        std::string aName =  aNC.NameRes("CB");

     std::cout << "BBBBB " << aName << "\n";
        Tiff_Im aFile
                (
                   aName.c_str(),
                   aSz,
                   aNC.TypeOut(false,aSFR),
                   Tiff_Im::No_Compr,
                   Tiff_Im::RGB,
                   anArg.ArgMTD()
                );
         Fonc_Num aMasqV =    aNC.ChannelFromName(NV1).MasqChannel()
	                   +  aNC.ChannelFromName(NV2).MasqChannel();
         GenIm::type_el aType = aNC.TypeOut(false,aSFR);

         Fonc_Num fR = TraitBasic ? aV3[0]->in_proj()  : anIm.in_proj();
         Fonc_Num fV = TraitBasic ? aV3[1]->in_proj()  : anIm.in_proj();
         Fonc_Num fB = TraitBasic ? aV3[2]->in_proj()  : anIm.in_proj();
         Fonc_Num aMR = TraitBasic ? 1 : aNC.ChannelFromName(NR).MasqChannel() * 4;
         Fonc_Num aMV = TraitBasic ? 1 : aMasqV * (0.5*4);
         Fonc_Num aMB = TraitBasic ? 1 : aNC.ChannelFromName(NB).MasqChannel()*4;


         fR = GamCor(anArg,(fR- aOfs[0]) * aMR * aWB[0],aNameFile);
         fV = GamCor(anArg,(fV- aOfs[1]) * aMV * aWB[1],aNameFile);
         fB = GamCor(anArg,(fB- aOfs[2]) * aMB * aWB[2],aNameFile);
/*
         Fonc_Num fR = GamCor(anArg,(anIm.in_proj()- aOfs[0]) * aNC.ChannelFromName(NR).MasqChannel() * aWB[0]*4);
         Fonc_Num fV = GamCor(anArg,(anIm.in_proj()- aOfs[1])  * 0.5 * aMasqV * aWB[1]*4);
         Fonc_Num fB = GamCor(anArg,(anIm.in_proj()- aOfs[2]) * aNC.ChannelFromName(NB).MasqChannel()*aWB[2]*4);
*/

         if (RBswap)
            ElSwap(fR,fB);

         Fonc_Num aF = Virgule(fR,fV,fB);
         if (!TraitBasic)
         {
             aF =  som_masq(aF,aPds);
         }

         if (HasFlF)
         {
              aF = aF / aFlF.in_proj()[Virgule(FX,FY)/10.0];
         }
         ELISE_COPY
	 (
	     aV3[0]->all_pts(),
	     Tronque (aType,aF),
	     aFile.out()
	 );

    }  

    std::vector<double> aCL = anArg.ClB();
    if (aCL.size() && (! TraitBasic))
    {
         Fonc_Num aMasqV =    aCL[0] * aNC.ChannelFromName(NR).MasqChannel()
	                   +  aCL[1] * aNC.ChannelFromName(NV1).MasqChannel()
	                   +  aCL[2] * aNC.ChannelFromName(NV2).MasqChannel()
	                   +  aCL[3] * aNC.ChannelFromName(NB).MasqChannel();

        double aMoyOf = (aOfs[0]*aCL[0] + aOfs[1]*(aCL[1]+aCL[2]) + aOfs[2]*aCL[3]) / (aCL[0]+aCL[1]+aCL[2]+aCL[3]);

        std::string aName =  aNC.NameRes(anArg.NameCLB());
        Tiff_Im aFile
                (
                   aName.c_str(),
                   anIm.sz(),
                   aNC.TypeOut(true,aSFR),
                   Tiff_Im::No_Compr,
                   Tiff_Im::BlackIsZero,
                   anArg.ArgMTD()
                );
         ELISE_COPY
	 (
	     anIm.all_pts(),
	     Tronque(aNC.TypeOut(true,aSFR),som_masq((anIm.in_proj()-aMoyOf)*aMasqV ,aPds)),
	     aFile.out()
	 );

    }

    if (! aSFR)
       ELISE_fp::RmFile(aNameTmp);

   if (anArg.BayerCalib()  && (! TraitBasic))
   {
      std::cout << "MpDcraw :: USE BAYER CALIBRATION \n";
      std::vector<double> aPG = anArg.PG();


      if (anArg.BayerCalib()->PG().IsInit() && (! anArg.PGSpec()))
      {
         Pt3dr aPG2 = anArg.BayerCalib()->PG().Val();
	 aPG[0] = aPG2.x;
	 aPG[1] = aPG2.y;
	 aPG[2] = aPG2.z;
      }

      double aPGT  = aPG[0]+ aPG[1] +  aPG[2];
      aPG[0] /= aPGT;
      aPG[1] /= aPGT;
      aPG[2] /= aPGT;



      for (int aKC=0; aKC<int(aNC.mVCh.size()) ; aKC++)
      {
          aNC.mVCh[aKC].InitParamGeom
	  (
	       anArg.MastCh(),
	       anArg.ScaleMast(),
	       anArg.BayerCalib(),
	       anArg.Interpol()
	  );
      }
      for (int aKC=0; aKC<int(aNC.mVCh.size()) ; aKC++)
      {
          aNC.mVCh[aKC].MakeInitImReech();
      }


      aNC.mSzR = aNC.ChannelFromName(NR).ImReech().sz();
      Fonc_Num fR = aNC.ChannelFromName(NR).ImReech().in_proj();
      Fonc_Num fV = (
	                      aNC.ChannelFromName(NV1).ImReech().in_proj()
	                    + aNC.ChannelFromName(NV2).ImReech().in_proj()
                       ) *0.5;
      Fonc_Num fB = aNC.ChannelFromName(NB).ImReech().in_proj();
      GenIm::type_el aType = aNC.TypeOut(false,aSFR);

      if (RBswap)
         ElSwap(fR,fB);

      if (anArg.ColReech())
      {
          std::string aName =  aNC.NameRes("CR");

          Tiff_Im aFile
                (
                   aName.c_str(),
                   aNC.mSzR,
                   aType,
                   Tiff_Im::No_Compr,
                   Tiff_Im::RGB,
                   anArg.ArgMTD()
                );

         Fonc_Num aFR = (fR-aOfs[0])*aWB[0];
         Fonc_Num aFV = (fV-aOfs[1])*aWB[1];
         Fonc_Num aFB = (fB-aOfs[2])*aWB[2];

         if (anArg.Adapt8B())
         {
             double aVM;
             ELISE_COPY(aFile.all_pts(),FilterMax(anArg,aFR+aFV+aFB,aFile),VMax(aVM));

             aVM /= 3.0; 
             double aMul = ElMin(1.0,255.0/aVM);
             aFR = (fR-aOfs[0])*(aWB[0]*aMul);
             aFV = (fV-aOfs[1])*(aWB[1]*aMul);
             aFB = (fB-aOfs[2])*(aWB[2]*aMul);
         }

/*
         aF = anArg.Adapt8B() ?
                       aITMP.in() * (255.0/aVMax) :
                       aITMP.in();
*/

	 ELISE_COPY
	 (
	     aFile.all_pts(),
	     Virgule
	     (
	           Tronque(aType,GamCor(anArg,aFR,aNameFile)),
	           Tronque(aType,GamCor(anArg,aFV,aNameFile)),
	           Tronque(aType,GamCor(anArg,aFB,aNameFile))
	     ),
	     aFile.out()
	 );
      }
      if (anArg.GrayReech())
      {
          std::string aName =  aNC.NameRes("GR");

          Tiff_Im aFile
                (
                   aName.c_str(),
                   aNC.mSzR,
                   aType,
                   Tiff_Im::No_Compr,
                   Tiff_Im::BlackIsZero,
                   anArg.ArgMTD()
                );

         Im2D_REAL4  aITMP(aNC.mSzR.x,aNC.mSzR.y);
         double aVMax;
         Fonc_Num aF = ((fR-aOfs[0])*aWB[0]*aPG[0]+(fV-aOfs[1])*aWB[1]*aPG[1]+(fB-aOfs[2])*aWB[2]*aPG[2]);
         aF = GamCor(anArg,aF,aNameFile); 
         ELISE_COPY
	 (
	     aITMP.all_pts(),
	     Virgule(aF, FilterMax(anArg,aF,aFile)),
	     Virgule(aITMP.out() , VMax(aVMax))
	 );

         aF = anArg.Adapt8B() ?
                       aITMP.in() * (255.0/aVMax) :
                       aITMP.in();

	 ELISE_COPY
	 (
	     aFile.all_pts(),
	     Tronque
	     (
	        aType,
		aF
	     ),
	     aFile.out()
	 );
      }
   }

   if (anArg.Diag()  && (! TraitBasic))
   {
      aNC.MakeImageDiag(anIm,"V","W",anArg);
   }

   return aNC;
}

std::string cNChannel::NameRes(const std::string & aName,const std::string & aPref) const
{
   return mArg.NameRes(mNameFile,aName,aPref) ;
/*
   std::string aN1 = StdPrefix(mNameFile);

   std::string aExtA =  mArg.ExtensionAbs();
   if (aExtA!="")
   {
           if (aExtA=="None") aExtA="";
       return mNameDir +aN1+aExtA+".tif";
   }

   std::string aN2 =  mArg.Extension();
 
   if (mArg.Add16_8B())
       aN2 = aN2+ (mArg.Cons16B()? "16B" : "8B") ;

   if ( mArg.NameOriIsPrefix())
      ElSwap(aN1,aN2);

   std::string aRes =  mNameDir 
         + aPref
         + aN1
         + std::string("_") 
	 + aN2
         + std::string("_") 
	 + aName 
	 +  std::string(".tif");

   std::cout << aRes << " "<< mArg.NameRes(mNameFile,aName,aPref) << "\n";

   return aRes;
*/
}


void cNChannel::SauvInit()
{
   for (int aK=0 ; aK<int(mVCh.size()) ; aK++)
   {
      mVCh[aK].SauvInit ();
   }
}

const cArgMpDCRaw & cNChannel::Arg() const
{
    return mArg;
}

GenIm::type_el  cNChannel::TypeOut(bool Signed,cSpecifFormatRaw * aSFR) const
{
   bool B16 = Arg().Cons16B();
   if (aSFR) B16 = aSFR->NbBitsParPixel() >=16;


   if (Signed)
       return B16 ? GenIm::int2 : GenIm::int1;

   return B16 ? GenIm::u_int2 : GenIm::u_int1;
}


cOneChanel &    cNChannel::ChannelFromName(const std::string & aName)
{
   for (int aK=0; aK<int(mVCh.size()) ; aK++)
      if (mVCh[aK].Name()== aName)
         return mVCh[aK];
   ELISE_ASSERT(false,"cNChannel::ChannelFromName");
   return * ((cOneChanel *) 0);
}
 

Pt2di cNChannel::I2Diag(const Pt2di & aP) const
{
    Pt2di  aQ0 = aP -mP0Diag;
    Pt2di  aQ1(
                   (  aQ0.x          -  mSDiag *aQ0.y)/2,
                   (  mSDiag*aQ0.x   +          aQ0.y)/2
              );

    return aQ1 - mOfsI2Diag;
    
}

Pt2di  cNChannel::Diag2I(const Pt2di & aP) const
{
     Pt2di  aQ0 = aP+mOfsI2Diag;
     Pt2di  aQ1  ( 
                    aQ0.x           +  mSDiag *aQ0.y,
                    -mSDiag*aQ0.x   +          aQ0.y
                 );

    return aQ1+mP0Diag;
}


void cNChannel::MakeImageDiag(Im2D_REAL4 aFulI,const std::string & aN1,const std::string & aN2,const cArgMpDCRaw & anArg)
{
    cOneChanel & aCh1 = ChannelFromName(aN1);
    cOneChanel & aCh2 = ChannelFromName(aN2);
    mP0Diag = aCh1.P0();
    mOfsI2Diag = Pt2di(0,0);
    mSDiag = 1;

    {
       ELISE_ASSERT(aCh1.Per()==Pt2di(2,2),"Per1 cNChannel::MakeImageDiag");
       ELISE_ASSERT(aCh2.Per()==Pt2di(2,2),"Per2 cNChannel::MakeImageDiag");

       Pt2di aP0b = aCh2.P0();
       ELISE_ASSERT(ElAbs(mP0Diag.x-aP0b.x)==1,"X-Phasage in cNChannel::MakeImageDiag");
       ELISE_ASSERT(ElAbs(mP0Diag.y-aP0b.y)==1,"Y-Phasage in cNChannel::MakeImageDiag");
    }

    Pt2di aSzIn = aFulI.sz();
    Box2di aBoxIn(Pt2di(0,0),aSzIn);
    Pt2di aCorIn[4];
    aBoxIn.Corners(aCorIn);
  
    Pt2di aP0Out = I2Diag(aCorIn[0]);
    Pt2di aP1Out = I2Diag(aCorIn[0]);
    for (int aK=0 ; aK<4 ;aK++)
    {
       aP0Out.SetInf(I2Diag(aCorIn[aK]));
       aP1Out.SetSup(I2Diag(aCorIn[aK]));
    }

    mOfsI2Diag = aP0Out;
    mSzDiag = aP1Out-aP0Out;


   Im2D_REAL4 aIOut(mSzDiag.x,mSzDiag.y);

   TIm2D<float,double> aTIn(aFulI);
   TIm2D<float,double> aTOut(aIOut);

   Pt2di aPOut;
   for (aPOut.x=0 ; aPOut.x<mSzDiag.x ; aPOut.x++)
   {
       for (aPOut.y=0 ; aPOut.y<mSzDiag.y ; aPOut.y++)
       {
           aTOut.oset(aPOut,1+aTIn.get(Diag2I(aPOut),-1));
       }
   }

   std::string aName =  NameRes("Diag");
   GenIm::type_el aType = TypeOut(false,0);
   Tiff_Im aFile
                (
                   aName.c_str(),
                   mSzDiag,
                   aType,
                   Tiff_Im::No_Compr,
                   Tiff_Im::BlackIsZero,
                   anArg.ArgMTD()
                );
   ELISE_COPY(aIOut.all_pts(),Tronque(aType,aIOut.in()),aFile.out());

}


CamStenope * cNChannel::CamCorDist() const
{
   return mCamCorDist;
}

cElHomographie *   cNChannel::HomograRedr() const
{
    return mHomograRedr;
}


cElHomographie *   cNChannel::InvHomograRedr() const
{
    return mInvHomograRedr;
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
