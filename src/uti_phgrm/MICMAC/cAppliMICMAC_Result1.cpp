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
#ifdef MAC
// Modif Greg pour avoir le nom de la machine dans les log
#include <sys/utsname.h>
#endif


static Pt2di aSzTileMasq(1000000,1000000);

bool cAppliMICMAC::DoNotTA() const
{

   return      DoNothingBut().IsInit()
	   && (! ButDoTA().Val());

}

/*
    int aDeZoom = ElMax(0,round_ni(log2(aSzGlob.x * aSzGlob.y / 3.0e7)));

   if (ZoomMakeTA().Val() <= 0) 
      return true;
      */

std::string cAppliMICMAC::NameEtatAvancement()
{
    return  FullDirResult()
            + std::string("MM_Avancement_")
            + mNameChantier
            + std::string(".xml");
}

cMM_EtatAvancement cAppliMICMAC::EtatAvancement()
{
    std::string aNameAv = NameEtatAvancement();
    if (ELISE_fp::exist_file(aNameAv))
    {
        return StdGetObjFromFile<cMM_EtatAvancement>
               (
                   aNameAv,
                   StdGetFileXMLSpec("ParamMICMAC.xml"),
                   "MM_EtatAvancement",
                   "MM_EtatAvancement"
               );
    }

    cMM_EtatAvancement aRes ;
    aRes.AllDone() = false;

    return aRes;
}

void cAppliMICMAC::SauvEtatAvancement(bool AllDone)
{
   if (CalledByProcess().Val())
      return;

   cMM_EtatAvancement aRes;
   aRes.AllDone() = AllDone;
   MakeFileXML(aRes,NameEtatAvancement());
}
 


void cAppliMICMAC::MakeFileTA()
{
    if (DoNotTA()) 
       return;


    bool aSauvTA = true;
    int aDeZoom = ZoomMakeTA().ValWithDef(mDeZoomFilesAux);
    if (aDeZoom <= 0)
    {
        aSauvTA = false;
	aDeZoom = mDeZoomFilesAux;
    }

    
    std::string aPrefTA =   FullDirResult()
                          + std::string("TA_")
                          + mNameChantier;

    std::string aNameTA = aPrefTA + std::string(".tif");

    if (
          (ELISE_fp::exist_file(aNameTA))
       && (mMemPart.NbMaxImageOn1Point().IsInit())
       )
       return;

// TA1
   {
       bool Done=false;
       for
       (
            tContEMC::const_iterator itE = mEtapesMecComp.begin();
            itE != mEtapesMecComp.end();
            itE++
       )
       {
          if ( (! Done) && ((*itE)->DeZoomTer() == aDeZoom))
          {
             Done = true;
             cFileOriMnt aFOM = OrientFromOneEtape(**itE);
             MakeFileXML(aFOM,aPrefTA+".xml");
          }
       }
   }
// TA1
    cGeomDiscFPx aGeomDFPx =  *mGeomDFPx;
    aGeomDFPx.SetDeZoom(aDeZoom);
    Pt2di aSzClip = aGeomDFPx.SzDz();
    std::cout << "aSzClip : " << aSzClip <<  " " << aDeZoom << "\n";
// getchar();
    Im2D_U_INT1 aImCpt(aSzClip.x,aSzClip.y,0);
    U_INT1 ** aDCpt = aImCpt.data();
    Im2D_U_INT1 aImLabel(aSzClip.x,aSzClip.y,255);
    U_INT1 ** aDataLabel = aImLabel.data();


    Im2D_U_INT1 aImGray(aSzClip.x,aSzClip.y,0);
    U_INT1 ** aDGr = aImGray.data();


    Pt2di aSzOr = OrthoTA().Val() ? aSzClip : Pt2di(1,1);
    Im2D_REAL4 aImPriOr(aSzOr.x,aSzOr.y,60000);
    REAL4 ** aDPI = aImPriOr.data();


    Im2D_Bits<1>  aImOKIm1(1,1);
    TIm2DBits<1>  aTOK1(aImOKIm1);
    if (mUseConstSpecIm1)
    {
          Tiff_Im aFIm = FileMasqOfResol(aDeZoom);
          aImOKIm1 = Im2D_Bits<1>(aSzClip.x,aSzClip.y);
          aTOK1 =  TIm2DBits<1>(aImOKIm1);
          ELISE_COPY(aFIm.all_pts(),aFIm.in(),aImOKIm1.out());
    }

   
    std::string nomAvancement = WorkDir()+std::string("avancement-")+NameChantier()+std::string(".txt"); 

    int num=0;
#ifdef MAC
    utsname buf;
    uname(&buf);
#endif
    int aLabel = 0;
    cSetName *  aSelector = mICNM->KeyOrPatSelector(FilterTA());

    int aNKB = TAUseMasqNadirKBest().ValWithDef(-1);



    for (tCsteIterPDV itFI=PdvBegin(); itFI!=PdvEnd(); itFI++)
    {
        if (aSelector->IsSetIn((*itFI)->Name()))
        {
           aLabel++;
           std::cout << "TA : "  << (*itFI)->Name() << "\n";
	   if (mWM) 
	   {
		std::ofstream fic(nomAvancement.c_str(),std::ios_base::app);
		//fic << "Etape 0 image "<<num<<" / "<< NbPdv()<< " : Done"<<std::endl;
#ifdef MEC
		fic << "<Top><Machine>"<<buf.nodename<<"</Machine><NumEtape>0</NumEtape><NbEtapes>5</NbEtapes><NumTache>"
                    << num << "</NumTache><NbTaches>"
                    << NbPdv() << "</NbTaches></Top>" << std::endl;
#else
		fic << "<Top><NumEtape>0</NumEtape><NbEtapes>5</NbEtapes><NumTache>"
                    << num << "</NumTache><NbTaches>"
                    << NbPdv() << "</NbTaches></Top>" << std::endl;
#endif 
		std::cout << "mWM : Preparation de l'image " << num << " / "<< NbPdv()<<std::endl;
	  }
          TIm2D<U_INT1,INT> aMasqNadir(Pt2di(1,1));
	  ++num;
          cGeomImageData  aGID = (*itFI)->Geom().SauvData();

          cInterfModuleImageLoader * anIMIL = (*itFI)->IMIL();
          Im2D_REAL4 aIm = CreateAllImAndLoadCorrel<Im2D_REAL4>(anIMIL,aDeZoom);
          TIm2D<REAL4,REAL8> aTIm(aIm);

          ELISE_COPY
          (
               aIm.all_pts(),
               ImFileForVisu(aIm,GammaVisu().Val()),
               aIm.out()
          );



          (*itFI)->Geom().SetDeZoomIm(aDeZoom);
	  const double * aPx0= aGeomDFPx.V0Px();
          Box2dr aBox = (*itFI)->Geom().EmpriseTerrain(aPx0,aPx0,100);

// std::cout << "ffgf" << aBox._p0 << aBox._p1 << "\n";
// getchar();

          Pt2di aP0 =  aGeomDFPx.R2ToDisc(aBox._p0);
          Pt2di aP1 =  aGeomDFPx.R2ToDisc(aBox._p1);
          pt_set_min_max(aP0,aP1);
          aP0.SetSup(Pt2di(0,0));
          aP1.SetInf(aSzClip);


          cGeomImage & aGeomIm = (*itFI)->Geom();


          ElSeg3D  aSegCentral(Pt3dr(0,0,0),Pt3dr(1,0,0)) ;
          Pt3dr aTNCentr;
          if (OrthoTA().Val())
          {
              CamStenope *  aCS = aGeomIm.GetOri();
              if (aCS)
              {
                   Pt2dr aPP =  aCS->PP();
                   aPP = aCS->DistDirecte(aPP);
                   // std::cout << "CSS  " <<  aPP <<  aCS->PP() << aCS->F2toDirRayonR3(aPP) <<  " " <<  aCS->F2toDirRayonL3(aPP) << "\n";
                   aTNCentr = aCS->F2toDirRayonR3(aPP);
              }
              else
              {
                 Pt2dr aPC = (aIm.sz()*aDeZoom)/2.0;
                 aSegCentral = aGeomIm.FaisceauPersp(aPC);
                 aTNCentr = aSegCentral.TgNormee();
              }
          }

         double aPdsT1 = 0.7;
         double aPdsT2 = 0.3;

          if (aNKB >= 0)
          {
               aMasqNadir= TIm2D<U_INT1,INT>(Im2D_U_INT1::FromFileStd(aGeomIm.NameMasqImNadir(aNKB)));
               // std::cout << "NNaddirr " << aMasqNadir.sz() << " " <<  aIm.sz() << "\n";
               ELISE_ASSERT(euclid(aMasqNadir.sz(),aIm.sz()) <2,"Taille incoherente dans masq nadir TA");
          }


          for (int aX = aP0.x; aX< aP1.x ; aX++)
          {
              for (int aY = aP0.y; aY< aP1.y ; aY++)
              {
                 if (!mUseConstSpecIm1 || aTOK1.get(Pt2di(aX,aY),0))
                 {
                      // Pt2dr aP = (*itFI)->Geom().CurObj2Im(aGeomDFPx.DiscToR2(Pt2di(aX,aY)),aPx0);
                      Pt2dr aP2Ter = aGeomDFPx.DiscToR2(Pt2di(aX,aY));
                      Pt2dr aP = aGeomIm.CurObj2Im(aP2Ter,aPx0);
                      // Pt2dr aP = aGeomIm.CurObj2Im(aGeomDFPx.DiscToR2(Pt2di(aX,aY)),aPx0);

                      bool Ok = aTIm.inside_rab(aP,3);
                      if (aNKB>=0) Ok = Ok && aMasqNadir.get(round_ni(aP),0);


                      if (Ok)
                      {
                         int aV = round_ni(aTIm.getr(aP,-1));

                         if (!OrthoTA().Val())
                         {
                            aDGr [aY][aX] = aV;
                            aDCpt[aY][aX]++;
                         }
                         else
                         {
                             Pt3dr aP3Ter1(aP2Ter.x,aP2Ter.y,aPx0[0]);
                             Pt3dr aP3Ter2(aP2Ter.x,aP2Ter.y,aPx0[0]-1e5);
                             Pt3dr aQTer1 = aGeomIm.GeomFinale2GeomFP(aP3Ter1);
                             Pt3dr aQTer2 = aGeomIm.GeomFinale2GeomFP(aP3Ter2);
                             Pt3dr aNormEucl = vunit(aQTer2-aQTer1);

                             ElSeg3D  aSegCur = aGeomIm.FaisceauPersp(aP*aDeZoom);
                             Pt3dr aTNCur = aSegCur.TgNormee();
                  
                             // double aTeta1 = acos(-aTNCur.z);
                             double aTeta1 = acos(scal(aTNCur,aNormEucl));
                             double aTeta2 = acos(scal(aTNCentr,aTNCur));

/*
std::cout << aPdsT1*aTeta1+aPdsT2*aTeta2 << "\n";
                             int aPri = ElMin(60000,round_ni((aPdsT1*aTeta1+aPdsT2*aTeta2)*50));
*/
                             double aPri = aPdsT1*aTeta1+aPdsT2*aTeta2;

                             if (aPri<aDPI[aY][aX])
                             {
                               aDPI[aY][aX] = aPri;
                               aDGr [aY][aX] = aV;
                               aDataLabel[aY][aX] = aLabel;
                             }
                             aDCpt[aY][aX]++;
                         }
                      }
                 }
              }
            }
            (*itFI)->Geom().RestoreData(aGID);
        }
    }
    if (! mMemPart.NbMaxImageOn1Point().IsInit())
    {
       int aMaxCpt;
       ELISE_COPY(aImCpt.all_pts(),aImCpt.in(),VMax(aMaxCpt));
       mMemPart.NbMaxImageOn1Point().SetVal(aMaxCpt);
       SauvMemPart();
    }


    if (aSauvTA)
    {
        double aSat = SaturationTA().Val();
     
        Fonc_Num aF = aImGray.in();
        if (aSat != 0)
        {
          aF = its_to_rgb(Virgule(aImGray.in(),(aImCpt.in()*19)%256,aSat));
        }


        Tiff_Im::Create8BFromFonc(aNameTA,aSzClip,aF);
    }

    if (Section_Results().MakeImCptTA().Val())
    {
        std::string aNameCpt = DirOfFile(aNameTA) + "Cpt" + NameWithoutDir(aNameTA);
        Tiff_Im::CreateFromIm(aImCpt,aNameCpt);
        if (OrthoTA().Val())
        {
             std::string aNameLabel = DirOfFile(aNameTA) + "Label" + NameWithoutDir(aNameTA);
             Tiff_Im::CreateFromIm(aImLabel,aNameLabel);
        }
    }
}


/*********************************************************/
/*                                                       */
/*     Gestion des images de masques                     */
/*                                                       */
/*********************************************************/

void cAppliMICMAC::VerifSzFile(Pt2di aSzF) const
{


   if ((aSzF.x==0) || (aSzF.y==0))
   {
    {   std::cout << "cAppliMICMAC::VerifSzFile \n"; getchar();}
      MicMacErreur
      (
         eErrImageFileEmpty,
         "Fichier image vide",
         "Zone de recouvrement insuffisante"
      );
   }
   
}

std::string cAppliMICMAC::NameImageMasqOfResol(int aDeZoom)
{
   // if (aDeZoom == 1) return WorkDir()+ImMasq().Val();

   return    FullDirMEC() 
          +  std::string("Masq_")
          +  mNameChantier
          +  std::string("_DeZoom")
          +  ToString(aDeZoom)
          +  std::string(".tif");

}

Pt2di  cAppliMICMAC::SzOfResol(int aDeZoom)
{
     cGeomDiscFPx aGeomDFPx =  *mGeomDFPx;
     aGeomDFPx.SetDeZoom(aDeZoom);
     VerifSzFile(aGeomDFPx.SzDz());

     return aGeomDFPx.SzDz();
}

Tiff_Im cAppliMICMAC::FileMasqOfResol(int aDeZoom) 
{
    std::string aNameMasq = NameImageMasqOfResol(aDeZoom);

    int aDZC = LazyZoomMaskTerrain().Val() ? mDeZoomMin : 1;
    if (! ELISE_fp::exist_file(aNameMasq))
    {
  
       if (aDeZoom==aDZC)
          MakeDefImMasq(aDZC);
       else
       {
           // cGeomDiscFPx aGeomDFPx =  *mGeomDFPx;
           // aGeomDFPx.SetDeZoom(aDeZoom);

           //std::string aNameMasqR2 = NameImageMasqOfResol(aDeZoom/2);
           FileMasqOfResol(aDeZoom/2);
           // VerifSzFile(aGeomDFPx.SzDz());
           std::string aNameRed = NameImageMasqOfResol(aDeZoom/2);
           std::cout  << "<< Make Masque for " <<  aNameRed << "\n";
           MakeTiffRed2BinaireWithCaracIdent
           (
                  aNameRed,
                  aNameMasq,
                  0.5,
                  SzOfResol(aDeZoom)
           );
           std::cout  << ">> Done Masque for " <<  aNameRed << "\n";
           double aRec = RecouvrementMinimal();
           if (aRec > 0)
           {
              Tiff_Im aTif = Tiff_Im::BasicConvStd(aNameMasq);
              double aNbOk;
              ELISE_COPY
              (
                   aTif.all_pts(),
                   Rconv(aTif.in()),
                   sigma(aNbOk)
              );
              Pt2di aSz = aTif.sz();
              if (aNbOk < round_ni((aRec*aSz.x)*aSz.y))
              {
                 std::cout 
                       << " GOT : " << aNbOk / double(aSz.x*aSz.y)
                       << " SPEC : " << aRec  << "\n";
                 MicMacErreur
                 (
                    eErrRecouvrInsuffisant,
                    "Recouvrement insuffisant",
                    "Specification de recouvrement utilisateur"
                 );
              }
           }
       }
    }

    return Tiff_Im::StdConvGen(aNameMasq,1,true,false);
}

Fonc_Num cAppliMICMAC::FoncSsPIMasqOfResol(int aDz)
{
if(0)
{
   Tiff_Im aTF =  FileMasqOfResol(aDz);
   Video_Win aW  = Video_Win::WStd(aTF.sz(),1.0);
   ELISE_COPY(aW.all_pts(), aTF.in(),aW.ogray());
   std::cout << "Donne Grayyy " << aTF.name() << "\n"; getchar();

   ELISE_COPY(aW.all_pts(), aTF.in_bool_proj(),aW.odisc());
   std::cout << "Donne disc \n"; getchar();


}
   return   FileMasqOfResol(aDz).in_bool_proj();
}

Fonc_Num cAppliMICMAC::FoncMasqOfResol(int aDz)
{
   Fonc_Num aFRes =  FileMasqOfResol(aDz).in_bool_proj();

   cCaracOfDeZoom * aCarac = GetCaracOfDZ(aDz);
   if (aCarac->HasMasqPtsInt())
   {
       Tiff_Im aFInt = Tiff_Im::BasicConvStd(aCarac->NameMasqInt());
       aFRes = aFRes && aFInt.in(0);
   }

   return aFRes;
}


// template <class tContPts> ElList<Pt2di> ToListPt2di(const tContPts & aCont)




void cAppliMICMAC::MakeDefImMasq(int aDeZoomCible)
{
/*
    if (Planimetrie().IsInit() && MasqueAutoByTieP().IsInit())
    {
       DoMasqueAutoByTieP();
    }
*/


    int aNbImMin = NbMinImagesVisibles().Val();
    std::string aNameMasq = NameImageMasqOfResol(aDeZoomCible);
    if (ELISE_fp::exist_file(aNameMasq))
       return;

    cGeomDiscFPx aGeomDFPx =  *mGeomDFPx;
    // aGeomDFPx.SetDeZoom(1);
    // Pt2di aSzGlob = aGeomDFPx.SzDz();
    // int aDeZoom = ElMax(0,round_ni(log2(aSzGlob.x * aSzGlob.y / 3.0e7)));
    // aDeZoom = 1<< aDeZoom;
    int aDeZoom = ZoomMakeMasq().ValWithDef(mDeZoomFilesAux);


    aGeomDFPx.SetDeZoom(aDeZoom);
    Pt2di aSzClip = aGeomDFPx.SzDz();

    Im2D_Bits<4>  aImCpt(aSzClip.x,aSzClip.y,0);
    TIm2DBits<4>  aTImCpt(aImCpt);

    Im2D_Bits<1>  aImOKIm1(aSzClip.x,aSzClip.y,0);
    TIm2DBits<1>  aTImOKIm1(aImOKIm1);
    bool FirstIm=true;


    bool aFullIm1= false;

    if (mFullIm1 && ChantierFullMaskImage1().Val())
    {
       ELISE_COPY(aImCpt.all_pts(),aNbImMin+1,aImCpt.out());
       ELISE_COPY(aImOKIm1.all_pts(),1,aImOKIm1.out());
       aFullIm1 =true;
    }
    else
    {
       if (! SingulariteInCorresp_I1I2().Val())
       {
          for (tCsteIterPDV itFI=PdvBegin(); itFI!=PdvEnd(); itFI++)
          {
             const std::vector<Pt2dr> & aVPZ1 = (*itFI)->Geom().ContourTer();
             ElList<Pt2di> aLPt;
             for (int aK=0; aK<int(aVPZ1.size()) ; aK++)
             {
                 Pt2di  aP(aGeomDFPx.R2ToRDisc(aVPZ1[aK]));
                 aLPt = aLPt+aP;
	      // std::cout  << aDeZoom << " " << aP << "\n";
             }
             ELISE_COPY
             (
                 polygone(aLPt),
                 Min(15,1+aImCpt.in(0)),
                 aImCpt.oclip()
             );
          // AJOUTER_OKIM1();
             if (FirstIm)
             {
                ELISE_COPY
                (
                    polygone(aLPt),
                    1,
                    aImOKIm1.oclip()
                );
             }
             FirstIm = false;
          }
       }
       else
       {
          for (tCsteIterPDV itFI=PdvBegin(); itFI!=PdvEnd(); itFI++)
          {
             cGeomImageData  aGID = (*itFI)->Geom().SauvData();
          // Tiff_Im aFile = (*itFI)->FileOfResol(aDeZoom);
              Pt2di aSz = Std2Elise((*itFI)->IMIL()->Sz(aDeZoom));
             Box2dr aBoxIm(Pt2dr(0,0),Pt2dr(aSz));

             (*itFI)->Geom().SetDeZoomIm(aDeZoom);
              const double * aPxC = aGeomDFPx.V0Px();
          // Box2dr aBox = (*itFI)->Geom().EmpriseTerrain(aPxC,aPxC,0);
          // Pt2di aP0 =  aGeomDFPx.R2ToDisc(aBox._p0);
          // Pt2di aP1 =  aGeomDFPx.R2ToDisc(aBox._p1);
          // pt_set_min_max(aP0,aP1);
          // aP0.SetSup(Pt2di(0,0));
          // aP1.SetInf(aSzClip);
	  //


	  //  On ne fait pas confiance aux calcul inverse
	     Pt2di aP0 = Pt2di(0,0);
	     Pt2di aP1 = aSzClip;

             for (int aX = aP0.x; aX< aP1.x ; aX++)
             {
                 for (int aY = aP0.y; aY< aP1.y ; aY++)
                 {
                      Pt2di aPTer(aX,aY);
                      Pt2dr aPIm = (*itFI)->Geom().CurObj2Im(aGeomDFPx.DiscToR2(aPTer),aPxC);

                      if (aBoxIm.inside(aPIm))
                      {
                         aTImCpt.oset(aPTer,ElMin(15,1+aTImCpt.get(aPTer)));
                         if (FirstIm)
                         {
                             aTImOKIm1.oset(aPTer,1);
                         }
                      }
                    
                 }
             }
             (*itFI)->Geom().RestoreData(aGID);
             FirstIm= false;
          }
       }
    }


/*
*/
    aGeomDFPx.SetDeZoom(aDeZoomCible);


    GenIm::type_el aTypeMask = Xml2EL(TypeMasque().Val());

    Tiff_Im aFileMasq
            (
                aNameMasq.c_str(),
                aGeomDFPx.SzDz(),
                aTypeMask,
                Xml2EL(ComprMasque().Val()),
                // GenIm::bits1_msbf,
		// Tiff_Im::No_Compr,
                //Tiff_Im::Group_4FAX_Compr,
                Tiff_Im::BlackIsZero,
                Tiff_Im::Empty_ARG
		+ Arg_Tiff(Tiff_Im::ANoStrip())
                +  Arg_Tiff(Tiff_Im::AFileTiling(aSzTileMasq))
            );

    //Fonc_Num aF = aImCpt.in(0) >= aNbImMin-0.5;
    Fonc_Num aF = aImCpt.in_proj() >= aNbImMin-0.5;


    if (mHasOneModeIm1Maitre)
    {
       if (! aFullIm1)
          aF = aF && (aImOKIm1.in(0)>0.5);
    }
    if (Planimetrie().IsInit() && MasqueTerrain().IsInit())
    {
       cFileOriMnt anOriCible = OrientFromParams(aDeZoom,1);
       Tiff_Im aFileMT = Tiff_Im::StdConvGen(WorkDir()+MasqueTerrain().Val().MT_Image(),1,true,true);

       Fonc_Num aFoncBasik = aFileMT.in_proj().v0();
       Im2D_Bits<1> aIMnt(1,1);
       if (aFileMT.mode_compr()!=Tiff_Im::No_Compr)
       {
           Pt2di aSz = aFileMT.sz();
           aIMnt  = Im2D_Bits<1>(aSz.x,aSz.y);
           ELISE_COPY(aFileMT.all_pts(),aFoncBasik!=0,aIMnt.out());
           aFoncBasik = aIMnt.in_proj();

       }


       Fonc_Num aFoncMT  = AdaptFoncFileOriMnt
                        (
                            anOriCible,
                            aFoncBasik,
                            WorkDir() + MasqueTerrain().Val().MT_Xml(),
                            false,
                            0
                        );
      aFoncMT = round_ni(aFoncMT);

    //Tiff_Im::Create8BFromFonc("MasqTerrainReech.tif",aImCpt.sz(),aFoncMT);


      aFoncMT = aFileMT.in_bool(aFoncMT);
      ELISE_COPY
      (
           select(aImCpt.all_pts(),aFoncMT==0),
           0,
          aImCpt.out()
      );
    }

   Fonc_Num aFoncMasq = aF[Virgule(FX,FY) * (aDeZoomCible/double(aDeZoom))];
   if (mUseConstSpecIm1)
   {
      Pt2di aSz = aFileMasq.sz();
      Im2D_Bits<1>  aImCont(aSz.x,aSz.y,0);
      ElList<Pt2di> aL = ToListPt2di(mContSpecIm1);
      ELISE_COPY(polygone(aL),1,aImCont.out());

     aFoncMasq = aFoncMasq && aImCont.in();
   }

    int aVMin,aVMax;
    min_max_type_num(aTypeMask,aVMin,aVMax);
    // std::cout << "HhhhhhhhhhhhhhhHHHh   "<< aVMax << "\n"; getchar();
    ELISE_COPY
    (
          aFileMasq.all_pts(),
          (aFoncMasq!=0) * (aVMax-1),
          aFileMasq.out() 
    );
    std::cout << ">>  Done Masq Resol 1 " << aNameMasq  <<  "\n";
}

cFileOriMnt cAppliMICMAC::OrientFromOneEtape(const cEtapeMecComp & anEtape) const
{
   cFileOriMnt aFOM;
   aFOM.Geometrie() = GeomMNT();
   mPDV1->Geom().RemplitOri(aFOM);
   anEtape.RemplitOri(aFOM); ;

   if (mGeomDFPx->TronkExport())
   {
       aFOM.mGXml.mPrec =  mMaxPrecision+1;
   }

   return aFOM;
}

cFileOriMnt cAppliMICMAC::OrientFromParams(int aDz,REAL aStepZ)
{
   // Je ne comprend pas pourquoi j'avais mis cette verif
   //  Apparament, elle n'a pas gene car c'est surtout Bati3D qui utilisait
   //  les masque terrains, toujours en DimPx 1 (il est vrai que , en
   //  geom image, le maque terrain peut etre confondu avec le masque im1 !)
   // ELISE_ASSERT(mDimPx==1,"cAppliMICMAC::OrientFromParams, dim!=1");

   cEtapeMecComp & anEt0 = **(mEtapesMecComp.begin());

   cGeomDiscFPx &   aGeom = anEt0.GeomTer();
   cGeomDiscFPx     aGeomInit = aGeom;

   aGeom.SetStep(&aStepZ);
   aGeom.SetDeZoom(aDz);

   cFileOriMnt aFOM = OrientFromOneEtape(anEt0);

   aGeom = aGeomInit;

   return aFOM;
}

/*
*/

cFileOriMnt cAppliMICMAC::GetOri(const std::string & aNameOri) const
{
   // MODIF MPD , utilisation de la fonction normale,
   // SVP ne pas modifier en cas de pb (au - ne pas commiter les modifs).
   // venir me voir ....
   return StdGetObjFromFile<cFileOriMnt>
          (
                 aNameOri,
                  StdGetFileXMLSpec("ParamChantierPhotogram.xml"),
                 "FileOriMnt",
                 "FileOriMnt"
          );
/*
    cElXMLTree aTree(aNameOri);

    {
       cElXMLTree aTreeSpec(mNameSpecXML);
       aTree.TopVerifMatch(&aTreeSpec,"FileOriMnt");
    }

    // Modif de Greg: si on donne directement la racine
    // le xml_init ne fonctionne pas (il cherche des noeuds a la profondeur 1 au lieu de 2)
    // plutot que modidier le xml_init, on cherche le FileOriMnt
    // A voir avec MPD
    cElXMLTree* brancheFileOriMnt=aTree.Get("FileOriMnt",1);

    cFileOriMnt  aOriInit;
*/
    // xml_init(aOriInit,brancheFileOriMnt/*&aTree*/);
    // return aOriInit;
}

Fonc_Num  cAppliMICMAC::AdaptFoncFileOriMnt
          (
                 const cFileOriMnt & anOriCible,
                 Fonc_Num            aFonc,
                 const std::string & aNameOri,
                 bool                aModifDyn,
                 double              aZOffset
          ) const
{

    return AdaptFonc2FileOriMnt
           (
                  "File = " + aNameOri,
                  anOriCible,
                  GetOri(aNameOri),
                  aFonc,
                   aModifDyn,
                  aZOffset,
                  Pt2dr(0,0)
            );
}




/*
Fonc_Num  cAppliMICMAC::AdaptFoncFileOriMnt
          (
                 const cFileOriMnt & anOriCible,
                 Fonc_Num            aFonc,
                 const std::string & aNameOri,
                 bool                aModifDyn,
                 double              aZOffset
          ) const
{
    cFileOriMnt  aOriInit = GetOri(aNameOri);

    if (! SameGeometrie(aOriInit,anOriCible))
    {
       std::cout << "File = [" << aNameOri << "]\n";
       std::cout  << "Geometrie incompatible Cible/Xml, AdaptFoncFileOriMnt\n" ;
    }

    
    Pt2dr aSxyC = anOriCible.ResolutionPlani();
    Pt2dr aSxyI = aOriInit.ResolutionPlani();

    double aSx = aSxyC.x / aSxyI.x;
    double aSy = aSxyC.y / aSxyI.y;

    if ((aSx<=0) || (aSy <=0))
    {
        std::cout << "File = [" << aNameOri << "]\n";
        ELISE_ASSERT
        (
            false,
            "Signe incompatibles dans AdaptFoncFileOriMnt"
        );
    }

    Pt2dr aTr = anOriCible.OriginePlani() -  aOriInit.OriginePlani();

    aFonc =  StdFoncChScale_Bilin
             (
                  aFonc,
                  Pt2dr(aTr.x/aSxyI.x,aTr.y/aSxyI.y),
                  Pt2dr(aSx,aSy)
             );

    if (aModifDyn)
    {
       double aZ0I = aOriInit.OrigineAlti();
       double aZ0C = anOriCible.OrigineAlti();
       double aStZC = anOriCible.ResolutionAlti();
       double aStZI = aOriInit.ResolutionAlti();
       aFonc = (aZOffset+ aZ0I-aZ0C)/aStZC + (aStZI/aStZC)*aFonc;
    }

    return aFonc;
}
*/



void cAppliMICMAC::GenereOrientationMnt()
{
   for
   (
        tContEMC::const_iterator itE = mEtapesMecComp.begin();
        itE != mEtapesMecComp.end();
        itE++
   )
   {
        GenereOrientationMnt(*itE);
   }
}

std::string   cAppliMICMAC::NameOrientationMnt(cEtapeMecComp * itE)
{
        return
		        FullDirMEC()
	              + std::string("Z_Num") 
                      + ToString((itE)->Num())
		      + std::string("_DeZoom")
		      + ToString((itE)->DeZoomTer())
		      + std::string("_")
		      + NameChantier()
                      + std::string(".xml");
}

void cAppliMICMAC::GenereOrientationMnt(cEtapeMecComp * itE)
{

    cFileOriMnt aFOM = OrientFromOneEtape(*itE);
    std::string aName =    NameOrientationMnt(itE);
    cElXMLTree * aTree = ToXMLTree(aFOM);
    FILE * aFP = ElFopen(aName.c_str(),"w");

         
    ELISE_ASSERT(aFP!=0,"cAppliMICMAC::GenereOrientationMnt");

    aTree->Show("      ",aFP,0,0,true);

    delete aTree;
    ElFclose(aFP);

    (itE)->DoRemplitXML_MTD_Nuage();

    GenTFW(aFOM,aName);
}


void GenTFW(const cFileOriMnt & aFOM,const std::string & aName)
{
     std::string aNameTFW = StdPrefix(aName) + ".tfw";
     std::ofstream aFtfw(aNameTFW.c_str());
     if (aFOM.mGXml.mPrec >=0)
        aFtfw.precision(aFOM.mGXml.mPrec);
     else
        aFtfw.precision(12);// modification jean christophe michelin je met une precision standard

     aFtfw << aFOM.ResolutionPlani().x << "\n" << 0 << "\n";
     aFtfw << 0 << "\n" << aFOM.ResolutionPlani().y << "\n";
     aFtfw << aFOM.OriginePlani().x << "\n" << aFOM.OriginePlani().y << "\n";
     aFtfw.close();
}

void GenTFW(const ElAffin2D & anAff,const std::string & aNameTFW)
{
    std::ofstream aFtfw(aNameTFW.c_str());
    aFtfw.precision(10);


    // Attention 1 Chance / 2 sur les terme croises aAfC2M.I10().y  et  aAfC2M.I01().x
    // GM: attention pour avoir un tfw valide il faut un champ par ligne
    aFtfw << anAff.I10().x << "\n" << anAff.I10().y << "\n";
    aFtfw << anAff.I01().x << "\n" << anAff.I01().y << "\n";
    aFtfw << anAff.I00().x << "\n" << anAff.I00().y << "\n";

    aFtfw.close();
}

double ResolOfAff(const ElAffin2D & anAff)
{
   return (euclid(anAff.I10()) +euclid(anAff.I01())) / 2.0;
}

double ResolOfNu(const cXML_ParamNuage3DMaille & aNu)
{
     ElAffin2D aAfM2C = Xml2EL(aNu.Orientation().OrIntImaM2C()); // RPCNuage
     return ResolOfAff(aAfM2C.inv());
}

Box2dr BoxTerOfNu(const cXML_ParamNuage3DMaille & aNu)
{
     Box2dr aRes(Pt2dr(0,0),Pt2dr(aNu.NbPixel()));

     ElAffin2D aAfM2C = Xml2EL(aNu.Orientation().OrIntImaM2C()); // RPCNuage
     return aRes.BoxImage(aAfM2C.inv());
}



void cAppliMICMAC::SauvParam()
{
  if  (!GenereXMLComp().Val())
     return;
 
  std::string aNameSaveOri = std::string(SYS_CP)+' '
							+std::string("\"")+mNameXML+std::string("\"")
							+ std::string(" \"")
							+ FullDirMEC()
							+ std::string("param_")
							+ mNameChantier
							+ std::string("_Ori.xml\"");

#if (ELISE_windows)
  // TODO: a generic function for copying files
  replace( aNameSaveOri.begin(), aNameSaveOri.end(), '/', '\\' );
#endif

  std::cout << aNameSaveOri << std::endl;
  VoidSystem(aNameSaveOri.c_str());

  cElXMLTree *  aComplParam = ToXMLTree(static_cast<cParamMICMAC&>(*this));
  std::string aNameCompl =  FullDirMEC() 
                          + std::string("param_")
                          + mNameChantier
                          + std::string("_Compl.xml");
  FILE * aFP2 = ElFopen(aNameCompl.c_str(),"w");
  ELISE_ASSERT(aFP2!=0,"cAppliMICMAC::Alloc File for _compl.xml");
  aComplParam->Show("      ",aFP2,0,0,true);
  ElFclose(aFP2);
  delete aComplParam;
}

void cAppliMICMAC::MakeFileFDC()
{
   if (! DoFDC().Val())
     return;

   cFileDescriptionChantier aFDC;
   std::list<cCouplesFDC> & lCple = aFDC.CouplesFDC();
   std::list<cImageFDC> & lIma = aFDC.ImageFDC();

   for (tCsteIterPDV iP1=PdvBegin(); iP1 != PdvEnd() ; iP1++)
   {
       const  cGeomImage & aGeo1Init = (*iP1)->Geom();
       Pt2dr aDirTr;
       bool aDoDirTr = aGeo1Init.DirEpipTransv(aDirTr);
       bool aMakeIm = aDoDirTr;

       if (aMakeIm)
       {
          cImageFDC aIm;
	  aIm.FDCIm() = (*iP1)->Name();
	  if (aDoDirTr)
	     aIm.DirEpipTransv().SetVal(aDirTr);
	  lIma.push_back(aIm);
       }

       // const  cGeomImage * aGeo1Ref = & aGeo1Init ;
       const  cGeomImage * aGeo1Ref = aGeo1Init.GeoTerrainIntrinseque();
       tCsteIterPDV iP2 = iP1;
       iP2++;
       for (; iP2 != PdvEnd() ; iP2++)
       {
           const  cGeomImage & aGeo2Init = (*iP2)->Geom();
           // const  cGeomImage * aGeo2Ref =  & aGeo2Init;
           const  cGeomImage * aGeo2Ref = aGeo2Init.GeoTerrainIntrinseque();
           Pt2dr aPTer;

//std::cout << "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAA\n";
           bool HasInter = aGeo1Ref->IntersectEmprTer(*aGeo2Ref,aPTer);
//std::cout << "BBBBBBBBBBBBbbbbbbbbbbbbbb\n";

           bool MakeCple = HasInter;

           if (MakeCple)
           {
               cCouplesFDC aCple;
               aCple.FDCIm1() = (*iP1)->Name();
               aCple.FDCIm2() = (*iP2)->Name();
               if (HasInter)
                  aCple.BSurH().SetVal(aGeo1Ref->BSurH(*aGeo2Ref,aPTer));
               lCple.push_back(aCple);
           }
       }
   }

  cElXMLTree *  aComplParam = ToXMLTree(aFDC);
  std::string aName= FullDirResult() + std::string("FDC.xml");
  FILE * aFP2 = ElFopen(aName.c_str(),"w");
  ELISE_ASSERT(aFP2!=0,"cAppliMICMAC::Alloc File for FDC.xml");
  aComplParam->Show("      ",aFP2,0,0,true);
  ElFclose(aFP2);
  delete aComplParam;
}


void cAppliMICMAC::TestPointsLiaisons
     ( 
          CamStenope * anOriRef,
          CamStenope * anOri,
          cGeomImage *  aGeom2 
          
     ) const
{

   int aZ = ZoomVisuLiaison().Val();

   if (aZ <=0)
      return;

   ElPackHomologue aPack = mPDV1->ReadPackHom(mPDV2);


   cInterfModuleImageLoader * anIMIL = mPDV1->IMIL();
   Im2D_REAL4 aIm = CreateAllImAndLoadCorrel<Im2D_REAL4>(anIMIL,aZ);

   Video_Win aW = Video_Win::WStd(aIm.sz(),1);

   ELISE_COPY
   (
       aW.all_pts(),
       ImFileForVisu(aIm,GammaVisu().Val()),
       aW.ogray()
   );
   for 
   (
       ElPackHomologue::tIter anIter = aPack.begin();
       anIter != aPack.end();
       anIter++
   )
   {
       double aD;
       Pt3dr aP = anOriRef->PseudoInter 
                  (
                     anIter->P1(),
                     *anOri,
                     anIter->P2(),
                     &aD
                  );
       Pt2dr aP1 = anIter->P1();
       Pt2dr aP2 = anIter->P2();
       double aD1 = euclid(aP1-anOriRef->R3toF2(aP));
       double aD2 = euclid(aP2-anOri->R3toF2(aP));

       cout << "Im1 " 
            << aP1
            // OO  << " Prof " << anOriRef.Prof(aP)
            << " Prof " << anOriRef->ProfondeurDeChamps(aP)
            << " " << aD1 << " " << aD2 << "\n";
       if  (aGeom2)
       {
          Pt2dr aPx = aGeom2->P1P2ToPx(aP1,aP2);
          std::cout 
                << " Im1 " <<  aP1
                << " Im2 " <<  aP2
                << " Px "  <<  1/aPx.x
                << " Px "  <<  aPx.y
                << " Px Norm "  <<  aPx.y / (aD1+aD2)
                << "\n";
       }
       aW.draw_circle_abs
       (
          anIter->P1()/aZ,
          20*ElMin(6.0,(aD1+aD2)/2),
          aW.pdisc()(P8COL::red)
       );
   }

    for 
    (
         std::list<Pt3dr>::const_iterator itPt=ListeTestPointsTerrain().begin();
         itPt!=ListeTestPointsTerrain().end();
         itPt++
    )
    {
         Ori3D_Std * anOLiLi = anOriRef->NN_CastOliLib();
         Pt3dr aPt = anOLiLi->carte_to_terr(*itPt);

         // OO Pt3dr aPt = anOriRef.carte_to_terr(*itPt);
         Pt2dr aSz =  Pt2dr(mPDV1->SzIm()) ;
         std::cout 
             << " Ter = " << *itPt
             // OO << " Im1 = " << anOriRef.to_photo(aPt)
             << " Im1 = " << anOriRef->R3toF2(aPt)
             // OO << " Im2 = " <<    anOri.to_photo(aPt)
             << " Im2 = " <<    anOri->R3toF2(aPt)
             << " \n "
             << " Pi-Im1 = " <<   aSz-anOriRef->R3toF2(aPt)
             << " Pi-Im2 = " <<   aSz-anOri->R3toF2(aPt)
             << "\n\n";
    }
   // getchar();
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
