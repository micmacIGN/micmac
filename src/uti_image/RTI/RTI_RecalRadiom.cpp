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

#include "RTI.h"


class cAppli_RecalRadio : public  cAppliWithSetImage
{
     public :
          cAppli_RecalRadio(int argc,char ** argv);

          std::string mNameI1;
          std::string mNameMaster;
          std::string mNameParam;
          Pt2di       mSz;
};

cAppli_RecalRadio::cAppli_RecalRadio(int argc,char ** argv) :
    cAppliWithSetImage(argc-1,argv+1,TheFlagNoOri)
{
    const cInterfChantierNameManipulateur::tSet * aSetIm = mEASF.SetIm();

    if (aSetIm->size() ==0)
    {
          ELISE_ASSERT(false,"No image in RTI_RR");
    }
    else if (aSetIm->size() > 1)
    {
         ExpandCommand(3,"",true);
         return;
    }

    ElInitArgMain
    (
         argc,argv,
         LArgMain()  << EAMC(mNameI1, "Name Image 1",  eSAM_IsPatFile)
                     << EAMC(mNameParam, "Name XML", eSAM_IsExistFile),
         LArgMain()  << EAM(mNameMaster, "Master",true,"Name of master image if != def master", eSAM_IsExistFile)
    );

   cAppli_RTI aRTIA(mNameParam,eRTI_OldRecal,mNameI1);


   cOneIm_RTI_Slave * aSl = aRTIA.UniqSlave();
   // cOneIm_RTI_Master * aMas = aRTIA.Master();

/*
   cOneIm_RTI * aMas = aRTIA.Master();
   if (EAMIsInit(&mNameMaster))
   {
       cAppli_RTI * aRTIA2 = new cAppli_RTI (mNameParam,mNameMaster);
       aMas =  aRTIA2->UniqSlave();
   }
*/

   //   Tiff_Im aTif2 = aSl->DoImReduite();
   //   Tiff_Im aTifMasq (aSl->NameMasqR().c_str());
   //   Tiff_Im aTif1 = aMas->DoImReduite();
   Tiff_Im aTif1 = aSl->FileImFull("");
   Tiff_Im aTifMasq = aSl->MasqFull();
   Tiff_Im aTif2("Mediane.tif");

   mSz = aTif2.sz();
   ELISE_ASSERT((aTifMasq.sz()==mSz) && (aTif1.sz()==mSz),"Sz incohe in RTIRecalRadiom");
 
   Im2D_REAL4 aI1(mSz.x,mSz.y);
   ELISE_COPY(aTif1.all_pts(),aTif1.in(),aI1.out());
   TIm2D<REAL4,REAL8> aTI1(aI1);

   Im2D_REAL4 aI2(mSz.x,mSz.y);
   ELISE_COPY(aTif2.all_pts(),aTif2.in(),aI2.out());
   TIm2D<REAL4,REAL8> aTI2(aI2);

   Im2D_Bits<1> aIM(mSz.x,mSz.y);
   ELISE_COPY(aTifMasq.all_pts(),aTifMasq.in(),aIM.out());
   TIm2DBits<1> aTIM(aIM);

   
   Im2D_Bits<1> aIMDil (mSz.x,mSz.y);

   ELISE_COPY
   (
        aIM.all_pts(),
        dilat_d4(aIM.in(0) && (aI2.in(1e9)>aRTIA.Param().SeuilSat().Val()) ,1),  
        aIMDil.out()
   );
   

   
   int aStep = 5;
   Pt2di aP;
   std::vector<double> aV1s2;
   for (aP.x =0 ; aP.x<mSz.x ; aP.x+= aStep)
   {
       for (aP.y =0 ; aP.y<mSz.y ; aP.y+= aStep)
       {
           if (aTIM.get(aP))
           {
                double aRatio = aTI1.get(aP) / ElMax(0.1,aTI2.get(aP));
                aV1s2.push_back(aRatio);
           }
       }
   }
   double aMed = MedianeSup(aV1s2);
   std::cout << "MEDIAN= " << aMed << "\n";
   Tiff_Im::CreateFromFonc 
   (
          "RATIO"+aSl->Name()+".tif",
          mSz, 
          (aI1.in() /(Max(0.1,aI2.in())*aMed)),
          GenIm::real4
    );
   
}


int RTIRecalRadiom_main(int argc,char ** argv)
{
     cAppli_RecalRadio anAppli(argc,argv);

     return EXIT_SUCCESS;
}


void cAppli_RTI::MakeImageMed(const Box2di & aBox,const std::string & aNameIm)
{
   std::cout << " cAppli_RTI::MakeImageMed " << aBox._p0 << "\n";
   Pt2di aSz = aBox.sz();

  std::vector<Im2D_REAL4 *> aVIm;
  std::vector<TIm2D<REAL4,REAL8> *> aVTIm;
  Im2D_REAL4 aMed(aSz.x,aSz.y);
  TIm2D<REAL4,REAL8>  aTMed(aMed);

   for (int aK=0 ; aK<int(mVIms.size()) ; aK++)
   {
       if ((! mVIms[aK]->IsMaster()) || (aNameIm!="ImDif"))
       {
           aVIm.push_back(new Im2D_REAL4(aSz.x,aSz.y));
           aVTIm.push_back(new TIm2D<REAL4,REAL8>(*(aVIm.back())));

           ELISE_COPY
           (
               aVIm.back()->all_pts(),
               trans(mVIms[aK]->FileImFull(aNameIm).in(),aBox._p0),
               aVIm.back()->out()
           );
       }
   }

   Pt2di aP;
   for (aP.x=0 ; aP.x<aSz.x ; aP.x++)
   {
       for (aP.y=0 ; aP.y<aSz.y ; aP.y++)
       {
           std::vector<double> aVVals;
           for (int aK=0 ; aK<int(aVIm.size()) ; aK++)
           {
               aVVals.push_back(aVTIm[aK]->get(aP));
           }
           aTMed.oset(aP,MedianeSup(aVVals));
       }
   }

   Tiff_Im aTifMed(mNameImMed.c_str());
   ELISE_COPY
   (
       rectangle(aBox._p0,aBox._p1),
       trans(aMed.in(),-aBox._p0),
       aTifMed.out()
   );
}


void cAppli_RTI::MakeImageMed(const std::string & aNameIm)
{
   // mNameImMed = "Mediane.tif";
   Tiff_Im aTif1 = mMasterIm->FileImFull("");

   Tiff_Im
   (
       mNameImMed.c_str(),
       aTif1.sz(),
       (aNameIm=="ImDif") ? GenIm::real4 : aTif1.type_el(),
       Tiff_Im::No_Compr,
       aTif1.phot_interp()
   );


   Pt2di aSz = aTif1.sz();


   int aBloc = 500;

   Pt2di aP0;

   for (aP0.x=0 ; aP0.x<aSz.x ; aP0.x+=aBloc)
   {
       for (aP0.y=0 ; aP0.y<aSz.y ; aP0.y+=aBloc)
       {
            Pt2di aP1 = Inf(aSz,aP0+Pt2di(aBloc,aBloc));
            MakeImageMed(Box2di(aP0,aP1),aNameIm);
       }
   }
   
}

int RTIMed_main(int argc,char **argv)
{
    std::string aNameParam;
    ElInitArgMain
    (
         argc,argv,
         LArgMain()  << EAMC(aNameParam, "Name XML", eSAM_IsExistFile),
         LArgMain()  
    );

   cAppli_RTI aRTIA(aNameParam,eRTI_Med,"");
   aRTIA.MakeImageMed("ImDif");
  
   return EXIT_SUCCESS;
}

/**********************************************************/
/**********************************************************/
/**********************************************************/

void cAppli_RTI::MakeImageGrad(const Box2di & aBox)
{
   std::cout << " cAppli_RTI::MakeImageGrad " << aBox._p0 << "\n";
   Pt2di aSz = aBox.sz();

   std::vector<Im2D_REAL4 *> aVIm;
   std::vector<TIm2D<REAL4,REAL8> *> aVTIm;
   std::vector<cOneIm_RTI *>         aVGlobIm;

   

   for (int aK=0 ; aK<int(mVIms.size()) ; aK++)
   {
       if (! mVIms[aK]->IsMaster()) 
       {
           aVIm.push_back(new Im2D_REAL4(aSz.x,aSz.y));
           aVTIm.push_back(new TIm2D<REAL4,REAL8>(*(aVIm.back())));

           ELISE_COPY
           (
               aVIm.back()->all_pts(),
               trans(mVIms[aK]->FileImFull("ImDif").in(),aBox._p0),
               aVIm.back()->out()
           );
           aVGlobIm.push_back(mVIms[aK]);
       }
   }
   Im2D_REAL4 aIGx(aSz.x,aSz.y);
   TIm2D<REAL4,REAL8>  aTGx(aIGx);
   Im2D_REAL4 aIGy(aSz.x,aSz.y);
   TIm2D<REAL4,REAL8>  aTGy(aIGy);


   cGenSysSurResol * aSys =0;
   int aNbObs = aVIm.size();
     // int aNbEq = (aDeg * (aDeg+1) ) /2;
   bool aL1 = true;
   int aNbUnknown = 2;
   if (aL1)
      aSys = new SystLinSurResolu(aNbUnknown,aNbObs);
   else
      aSys = new L2SysSurResol(aNbUnknown);


   Pt2di aP;
   for (aP.x=0 ; aP.x<aSz.x ; aP.x++)
   {
       for (aP.y=0 ; aP.y<aSz.y ; aP.y++)
       {
// std::cout << "AAAAAAA\n";
           aSys->SetPhaseEquation(0);
// std::cout << "BBBBBBBB\n";
           Pt2dr aPGlob = Pt2dr(aP+ aBox._p0);
           Pt3dr aPTer = OriMaster()->ImEtProf2Terrain(aPGlob,0.0);
           for (int aK=0 ; aK<int(aVIm.size()) ; aK++)
           {
               Pt3dr aC = aVGlobIm[aK]->CenterLum();
               Pt2dr aDir(aC.x-aPTer.x,aC.y-aPTer.y);
               aDir = vunit(aDir);
               double aCoeff[2];
               aCoeff[0] = aDir.x;
               aCoeff[1] = aDir.y;

               aSys->GSSR_AddNewEquation(1,aCoeff,aVTIm[aK]->get(aP),(double *)0);
           }

// std::cout << "CCCCCCC\n";
           Im1D_REAL8 aSol = aSys->GSSR_Solve(0);
           REAL8 * aDS = aSol.data();
           aTGx.oset(aP,aDS[0]);
           aTGy.oset(aP,aDS[1]);


// if (aP.x==aP.y) std::cout << "DSSS " << aDS[0] << " " << aDS[1] << "\n";
           aSys->GSSR_Reset(true);
// std::cout << "ZZZZZZZZZZZ\n";
       }
   }

   delete aSys;

// std::cout << "GGGGG " << mNameImGx << " " << mNameImGy << "\n";
   Tiff_Im aTifGx(mNameImGx.c_str());
   Tiff_Im aTifGy(mNameImGy.c_str());
   ELISE_COPY
   (
       rectangle(aBox._p0,aBox._p1),
       trans(Virgule(aIGx.in(),aIGy.in()),-aBox._p0),
       Virgule(aTifGx.out(),aTifGy.out())
   );

/*
static Video_Win  aWX = Video_Win::WStd(aBox.sz(),1.0);
static Video_Win  aWY = Video_Win::WStd(aBox.sz(),1.0);

ELISE_COPY(aWX.all_pts(),Max(0,Min(255,128+aIGx.in()*40)),aWX.ogray());
ELISE_COPY(aWY.all_pts(),Max(0,Min(255,128+aIGy.in()*40)),aWY.ogray());
aWX.clik_in();
*/
}

void cAppli_RTI::MakeImageGrad()
{
   // mNameImMed = "Mediane.tif";
   Tiff_Im aTif1 = mMasterIm->FileImFull("");

   Tiff_Im(mNameImGx.c_str(),aTif1.sz(),GenIm::real4 ,Tiff_Im::No_Compr,aTif1.phot_interp());
   Tiff_Im(mNameImGy.c_str(),aTif1.sz(),GenIm::real4 ,Tiff_Im::No_Compr,aTif1.phot_interp());

   Pt2di aSz = aTif1.sz();

   int aBloc = 500;
   Pt2di aP0;

   for (aP0.x=0 ; aP0.x<aSz.x ; aP0.x+=aBloc)
   {
       for (aP0.y=0 ; aP0.y<aSz.y ; aP0.y+=aBloc)
       {
            Pt2di aP1 = Inf(aSz,aP0+Pt2di(aBloc,aBloc));
            MakeImageGrad(Box2di(aP0,aP1));
       }
   }
   
}

int RTIGrad_main(int argc,char **argv)
{
    std::string aNameParam;
    ElInitArgMain
    (
         argc,argv,
         LArgMain()  << EAMC(aNameParam, "Name XML", eSAM_IsExistFile),
         LArgMain()  
    );

   cAppli_RTI aRTIA(aNameParam,eRTI_Grad,"");
   aRTIA.MakeImageGrad();
  
   return EXIT_SUCCESS;
}

/**********************************************************/
/**********************************************************/
/**********************************************************/

void cAppli_RTI::FiltrageGrad()
{
    Im2D_REAL4 aGx = Im2D_REAL4::FromFileStd(mNameImGx);
    Im2D_REAL4 aGy = Im2D_REAL4::FromFileStd(mNameImGy);
    TIm2D<REAL4,REAL8> aTGx(aGx);
    TIm2D<REAL4,REAL8> aTGy(aGy);
    Im2D_Bits<1> aMasq = MasqFromFile("MasqGlob.tif");
    // Im2D_Bits<1> aMasq = Master()->ImMasqFull();
    TIm2DBits<1> aTMasq(aMasq);
    Pt2di aSz = aGx.sz();

    Im2D_REAL4 aR1(aSz.x,aSz.y,0.0);
    Im2D_REAL4 aR2(aSz.x,aSz.y,0.0);

    ELISE_COPY(aMasq.all_pts(),erod_d8(aMasq.in(0),1), aMasq.out());

    for (int aK= 0 ; aK<1000 ; aK++)
    {
        double aAtt = 0.95;
        TIm2D<REAL4,REAL8> aTI1(aR1);
        TIm2D<REAL4,REAL8> aTI2(aR2);

        Pt2di aP;
        for (aP.x=0 ; aP.x<aSz.x; aP.x++)
        {
            for (aP.y=0 ; aP.y<aSz.y; aP.y++)
            {
                if (aTMasq.get(aP))
                {
                    double aV =    aAtt * aTI1.get(aP)
                                +  (
                                         aTGx.get(aP+Pt2di(-1, 0)) 
                                       + aTGy.get(aP+Pt2di( 0,-1)) 
                                       - aTGx.get(aP+Pt2di( 1, 0)) 
                                       - aTGy.get(aP+Pt2di( 0, 1)) 
                                   ) / 4.0;
                    aTI2.oset(aP,aV);
                }
            }
        }

        std::cout << "K=" << aK << "\n";
        if (((aK+1) %10) == 0)
        {
             Tiff_Im::CreateFromFonc("MNT-" + ToString(aK)+".tif",aSz,aR1.in(),GenIm::real4);
             Tiff_Im::CreateFromFonc("MNTX-" + ToString(aK)+".tif",aSz,aR1.in()-trans(aR1.in(0),Pt2di(1,0)),GenIm::real4);
             Tiff_Im::CreateFromFonc("MNTY-" + ToString(aK)+".tif",aSz,aR1.in()-trans(aR1.in(0),Pt2di(0,1)),GenIm::real4);
        }
        
        ElSwap(aR1,aR2);
    }
}

int RTIFiltrageGrad_main(int argc,char **argv)
{
    std::string aNameParam;
    ElInitArgMain
    (
         argc,argv,
         LArgMain()  << EAMC(aNameParam, "Name XML", eSAM_IsExistFile),
         LArgMain()  
    );

   cAppli_RTI aRTIA(aNameParam,eRTI_Grad,"");
   aRTIA.FiltrageGrad();
  
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
Footer-MicMac-eLiSe-25/06/2007*/
