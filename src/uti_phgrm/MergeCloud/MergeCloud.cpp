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

#include "MergeCloud.h"

// std::string DirQMPLy () {return "QM-Ply/";}

class cCmpResolMCSPOM
{
    public :
       bool operator ()(const tMCSom * aS1,const tMCSom * aS2) {return aS1->attr()->Resol() < aS2->attr()->Resol();}
};

cAppliMergeCloud::cAppliMergeCloud(int argc,char ** argv) :
   cAppliWithSetImage(argc-1,argv+1,0),
   mTheWinIm        (0),
   mFlagCloseN      (mGr.alloc_flag_arc()),
   mSubGrCloseN      (mSubGrAll,mFlagCloseN),
   mGlobMaxNivH     (-1),
   mVStatNivs       (MaxValQualTheo() +2),
/*
   mImGainOfQual    (BestQual() +1),
   mDataGainOfQual  (mImGainOfQual.data()),
   mNbImOfNiv       (BestQual() +1,0),
   mDataNION        (mNbImOfNiv.data()),
   mCumNbImOfNiv    (BestQual() +2,0),
   mDataCNION       (mCumNbImOfNiv.data()),
*/
   mNbImSelected    (0),
   mDoPly           (true),
   mDoPlyCoul         (false),
   mSzNormale         (-1),
   mNormaleByCenter   (false),
   mModeMerge         (eQuickMac),
   mDS                (1.0),
   mOffsetPly         (0,0,0),
   mSH(""),
   mDoublePrec        (false)
{
   // ELISE_fp::MkDirSvp(Dir()+DirQMPLy());

   mVStatNivs[eQC_Out].mGofQ          = 0;
   mVStatNivs[eQC_ZeroCohBrd].mGofQ   = 1/32.0;
   mVStatNivs[eQC_ZeroCoh].mGofQ      = 1/16.0;
   mVStatNivs[eQC_ZeroCohImMul].mGofQ = 1/8.0;
   mVStatNivs[eQC_GradFort].mGofQ     = 1/4.0;
   mVStatNivs[eQC_GradFaibleC1].mGofQ = 1/3.0;
   mVStatNivs[eQC_Bord].mGofQ         =  1/2.0;
   mVStatNivs[eQC_Coh1].mGofQ         =  1.0;
   mVStatNivs[eQC_GradFaibleC2].mGofQ =  1.0;
   mVStatNivs[eQC_Coh2].mGofQ         =  2.0;
   mVStatNivs[eQC_Coh3].mGofQ         =  4.0;
   
   std::string aPat,anOri;
   std::string aModeMerge = "QuickMac";
   ElInitArgMain
   (
        argc,argv,
        LArgMain()  << EAMC(aPat,"Full Directory (Dir+Pattern)",eSAM_IsPatFile)
                    << EAMC(anOri,"Orientation ", eSAM_IsExistDirOri),
        LArgMain()  << EAM(mFileParam,"XMLParam",true,"File Param, def = XML_MicMac/DefMergeCloud.xml")
                    << EAM(mDoPly,"DoPly",true,"Generate Ply of selected files (Def=true)")
                    << EAM(mDoPlyCoul,"PlyCoul",true,"Generated ply are in coul (Def=false)")
                    << EAM(mSzNormale,"SzNorm",true,"Parameters for normals creation")
                    << EAM(mNormaleByCenter,"NormByC",true,"Normale by Center")
                    << EAM(aModeMerge,"ModeMerge",true,"Mode Merge in enumerated values", eSAM_None,ListOfVal(eNbTypeMMByP,"e"))
                    << EAM(mDS,"DownScale",true,"DownScale used in computation (to compute names)")
                    << EAM(mOffsetPly,"OffsetPly",true,"Offset Ply for Nuage2Ply")
                    << EAM(mSH,"SH",true,"Set of Hom, Def=\"\"")
            		    << EAM(mDoublePrec,"64B",true,"To generate 64 Bits ply, Def=false, WARN = do not work properly with meshlab or cloud compare")
   );



   if (! EAMIsInit(&mFileParam))
   {
         mFileParam = Basic_XML_MM_File("DefMergeCloud.xml");
   }

   mParam = StdGetFromSI(mFileParam,ParamFusionNuage);

   if (EAMIsInit(&aModeMerge))
   {
       bool aModeHelp;
       StdReadEnum(aModeHelp,mModeMerge,aModeMerge,eNbTypeMMByP);
   }
   else
   {
      mModeMerge = mParam.ModeMerge();
   }

   mMMIN = cMMByImNM::ForGlobMerge(Dir(),mDS,aModeMerge);



   mPatMAP  = new cElRegex (mParam.ImageMiseAuPoint().ValWithDef(".*"),10);

   // ===  Creation des nuages 

   for (int aKS=0 ; aKS<int(cAppliWithSetImage::mVSoms.size()) ; aKS++)
   {
        cImaMM * anIma = cAppliWithSetImage::mVSoms[aKS]->attr().mIma;
        const std::string & aNameIm = anIma->mNameIm;
        cASAMG * anAttrSom = 0;

        bool InMAP = false;
        //  std::string aNameNuXml = NameFileInput(true,anIma,".xml");
        std::string aNameNuXml = mMMIN->NameFileXml(eTMIN_Depth,aNameIm);
        // Possible aucun nuage si peu de voisins et mauvaise config epip

        if (ELISE_fp::exist_file(aNameNuXml))
        {
             // std::string aNM = NameFileInput(true,anIma,"_Masq.tif");
             std::string aNM =  mMMIN->NameFileMasq(eTMIN_Depth,aNameIm);

             // std::string aNM = NameFileInput(true,"Masq"+anIma,".tif");
             Tiff_Im aTM(aNM.c_str());
             int aSom;
             ELISE_COPY(aTM.all_pts(),aTM.in(),sigma(aSom));
            // Possible nuage vide 
            if (aSom>100)
            {
                anAttrSom = new cASAMG(this,anIma);
                mVAttr.push_back(anAttrSom);
                tMCSom &  aSom = mGr.new_som(anAttrSom);
                mDicSom[anIma->mNameIm] = & aSom;
                mVSoms.push_back(&aSom);
                InMAP = IsInImageMAP(anAttrSom) ;
            }
        }

        std::cout << anIma->mNameIm  << (anAttrSom ? " OK " : " ## ") << " MAP " << InMAP << " XMl " << aNameNuXml << "\n";
   }
   cCmpResolMCSPOM aCmp;
   std::sort(mVSoms.begin(),mVSoms.end(),aCmp);

   // Mise au point
   if (mParam.TestImageDif().Val() && (mVSoms.size()==2))
   {
        mVAttr[0]->TestDifProf(*(mVAttr[1]));
   }
   // Calcul connexion
   CreateGrapheConx();


std::cout << "CCcccccccccccc\n";
   // Calcul image de quality + Stats
   for (int aK=0 ; aK<int(mVSoms.size()) ; aK++)
   {
       std::cout << mVSoms[aK]->attr()->IMM()->mNameIm << " Resol=" <<  mVSoms[aK]->attr()->Resol() << "\n";
       mVSoms[aK]->attr()->TestImCoher();
       int aNiv = mVSoms[aK]->attr()->MaxNivH();
       mVStatNivs[aNiv].mNbIm ++;
       mVStatNivs[aNiv].mRecTot ++;

       ElSetMax(mGlobMaxNivH,aNiv);
   }
   for (int aK=0 ; aK<int(mVSoms.size()) ; aK++)
   {
       tMCSom * aS1 =  mVSoms[aK];
       int aNiv1 =  aS1->attr()->MaxNivH();
       for (tArcIter itA = aS1->begin(mSubGrAll) ; itA.go_on() ; itA++)
       {
            int aNiv2 =  (*itA).s2().attr()->MaxNivH();
            int aNiv = ElMin(aNiv1,aNiv2);
            double aRec = (*itA).attr()->Rec();
            mVStatNivs[aNiv].mRecTot += aRec;
       }
   }



   for (int aQ = MaxValQualTheo() ; aQ>=0 ; aQ--)
   {
       cStatNiv * aSN = VData(mVStatNivs) + aQ;
       cStatNiv * aNSN = aSN+1;
       aSN->mCumNbIm  = aSN->mNbIm   + aNSN->mCumNbIm;
       aSN->mCumRecT  = aSN->mRecTot + aNSN->mCumRecT;
       if (aSN->mCumNbIm)
       {
           aSN->mRecMoy =   aSN->mCumRecT / aSN->mCumNbIm;
           aSN->mNbImPourRec =  aSN->mCumNbIm /  aSN->mRecMoy;
       }
   }

   if (1)
   {
       for (int aQ=0 ; aQ<=MaxValQualTheo() +1 ; aQ++)
       {
           cStatNiv * aSN = VData(mVStatNivs) + aQ;
           std::cout << "Nb Im With Qual >= " << aQ << " = " <<  aSN->mNbImPourRec  << " NbCum " << aSN->mCumNbIm << " \n";
       }
       std::cout << "MAX NIV GLOB " << mGlobMaxNivH << "\n";
   }
   


   int aNivMin = ElMin(eQC_Coh1,eQC_GradFaibleC2);
   if (mModeMerge ==eStatue)
   {
/*
        <eQC_GradFaibleC1 >   </eQC_GradFaibleC1>
         <eQC_Bord >           </eQC_Bord>
         <eQC_Coh1 >           </eQC_Coh1>
         <eQC_GradFaibleC2 >   </eQC_GradFaibleC2>
*/
      aNivMin= eQC_GradFaibleC2;
   }
   if (IsMacType(mModeMerge))
   {
      aNivMin=eQC_GradFaibleC1;
   }
   // aNivMin = eQC_GradFaibleC2;
   if (aNivMin>mGlobMaxNivH) {
       // jo 2018 09; tentative de faire foncitonne pims2ply sur image go pro
       std::cout << "Warning : NivMin = " << aNivMin << ", Niv Gob max is "<< mGlobMaxNivH << "\n";
       aNivMin=mGlobMaxNivH-1;
       std::cout << "I change value of NivMin to " << aNivMin <<"\n";
   }
   for (mCurNivSelSom=mGlobMaxNivH ; mCurNivSelSom>=aNivMin ; mCurNivSelSom--)
   {
       std::cout << "BEGIN NIV " <<  mCurNivSelSom << " " << eToString(eQualCloud(mCurNivSelSom)) << "\n"; 
       mCurNivElim = ElMin(mCurNivSelSom,int(eQC_Coh2));
       OneStepSelection();
       std::cout << "END NIV " << mCurNivSelSom << "\n"; 
   }

   // Export Ply
   std::list<std::string> aLComPly;
   for (int aK=0 ; aK<int(mVSoms.size()) ; aK++)
   {
        std::string aCom = mVSoms[aK]->attr()->ExportMiseAuPoint();
        if (aCom!="")
        {
           aLComPly.push_back(aCom);
           std::cout << "COM= " << aCom << "\n";
           // getchar();
           // System(aCom);
        }
   }
   cEl_GPAO::DoComInParal(aLComPly);
}


void cAppliMergeCloud::OneStepSelection()
{
   for (int aK=0 ; aK<int(mVSoms.size()) ; aK++)
   {
         mVSoms[aK]->attr()->InitNewStep(mCurNivElim);
   }
   int aNbImMin = round_up(pAramSurBookingIm()*mVStatNivs[mCurNivSelSom].mNbImPourRec);
   std::cout << "NB IM MIN " << aNbImMin  << "\n";

   bool Cont = true;
   while (Cont)
   {
      tMCSom * aBestSom = 0;
      double aBestQual = 0;

      // On recherche la meilleure image
      for (int aK=0 ; aK<int(mVSoms.size()) ; aK++)
      {
          tMCSom *  aSom =  mVSoms[aK];
          if (      ( aSom->attr()->IsCurSelectable())
                &&  ( aSom->attr()->NivSelected() != mCurNivSelSom)
                &&  (aSom->attr()->MaxNivH() >=mCurNivSelSom)
             )
          {
             double aQual = aSom->attr()->QualOfNiv();
             // On rajoute quoiqu'il arrive les images déja selectionne
             if (aSom->attr()->NivSelected() > mCurNivSelSom) 
             {
                  aQual = 1000 * aQual + 1e8;
             }
             if (aBestQual<aQual)
             {
                  aBestQual = aQual;
                  aBestSom = aSom;
             }
          }
      }


      if (aBestSom)
      {
           cASAMG * aBestA = aBestSom->attr();
           double aRatio = aBestA->NbOfNiv() / double( aBestA->NbTot());
           std::cout << "SOM GOT " << aBestA->IMM()->mNameIm << " New=" << aBestA->IsSelected() << " R=" <<   aRatio << "\n";

           //
           if (! aBestA->IsSelected())
           {
               double aSeuil = (mNbImSelected>aNbImMin)  ? mParam.HighRatioSelectIm().Val() : mParam.LowRatioSelectIm().Val() ;
               Cont = aRatio > aSeuil;
           }

           if (Cont)
           {
               if (! aBestA->IsSelected())
                  mNbImSelected++;
               aBestA->SetSelected(mCurNivSelSom,mCurNivElim,aBestSom);
               // std::cout << "One Image Selected " << aBestA->NivSelected() << "\n"; getchar();
           }

      }
      else
      {
           Cont = false;
      }
      if (! Cont)
      {
      }
   }

   for (int aK=0 ; aK<int(mVSoms.size()) ; aK++)
   {
         mVSoms[aK]->attr()->FinishNewStep(mCurNivElim);
   }
}

bool cAppliMergeCloud::IsInImageMAP(cASAMG* anAS)
{
   return mPatMAP->Match(anAS->IMM()->mNameIm);
}



Video_Win *  cAppliMergeCloud::TheWinIm(const cASAMG * anAS)
{
   if ((! mParam.SzVisu().IsInit()) || (! anAS->IsImageMAP()))
     return 0;

   if (mTheWinIm==0)
   {
       Pt2di aSzIm = anAS->Sz();
       Pt2di aSzW = mParam.SzVisu().Val();
       double aRx = aSzW.x / double(aSzIm.x);
       double aRy = aSzW.y / double(aSzIm.y);
       mRatioW = ElMin(aRx,aRy);
       aSzW = round_ni(Pt2dr(aSzIm)*mRatioW);

       mTheWinIm = Video_Win::PtrWStd(aSzW,true,Pt2dr(mRatioW,mRatioW));
       mTheWinIm->set_cl_coord(Pt2dr(0,0),Pt2dr(mRatioW,mRatioW));
       std::cout << "RATIOW " << mRatioW << "\n";
   }

   return mTheWinIm;
}

/*
std::string cAppliMergeCloud::NameFileInput(bool DownScale,const std::string & aNameIm,const std::string & aPost,const std::string & aPrefIn)
{
   switch (mModeMerge)
   {
       case eQuickMac :
       case eStatue :
       {

            std::string aDirLoc = (mModeMerge== eStatue) ? DirFusStatue() : DirFusMMInit();
            std::string aPref = (aPrefIn=="" )? "Depth" : aPrefIn;
            std::string PrefGlob =  DownScale ? "DownScale_NuageFusion-" : "NuageFusion-";
            return Dir() + aDirLoc   +  PrefGlob + aPref + aNameIm + aPost;
       }

       default :
       ;

   }
   ELISE_ASSERT(false,"cAppliMergeCloud::NameFileInput");
   return "";
}

std::string cAppliMergeCloud::NameFileInput(bool DownScale,cImaMM * anIma,const std::string & aPost,const std::string & aPref)
{
    return NameFileInput(DownScale,anIma->mNameIm,aPost,aPref);
}
*/


tMCSom * cAppliMergeCloud::SomOfName(const std::string & aName)
{
   std::map<std::string,tMCSom *>::iterator it = mDicSom.find(aName);

   if (it==mDicSom.end()) return 0;
   return it->second;
}

REAL8    cAppliMergeCloud::GainQual(int aNiv) const
{
    return mVStatNivs[aNiv].mGofQ;
}

tMCSubGr &   cAppliMergeCloud::SubGrAll()
{
   return mSubGrAll;
}

bool  cAppliMergeCloud::DoPlyCoul() const
{
   return mDoPlyCoul;
}

cMMByImNM *  cAppliMergeCloud::MMIN()
{
   return mMMIN;
}


int  cAppliMergeCloud::SzNormale() const {return mSzNormale;}
bool cAppliMergeCloud::NormaleByCenter() const {return mNormaleByCenter;}



//========================================================================================


int CPP_AppliMergeCloud(int argc,char ** argv)
{
    cAppliMergeCloud anAppli(argc,argv);


    return EXIT_SUCCESS;
}


bool   cAppliMergeCloud::HasOffsetPly() 
{
   return EAMIsInit(&mOffsetPly);
}
const Pt3dr & cAppliMergeCloud::OffsetPly() 
{
   ELISE_ASSERT(HasOffsetPly()," cAppliMergeCloud::OffsetPly");
   return mOffsetPly;
}

const bool   cAppliMergeCloud::Export64BPly()
{
	return mDoublePrec;
}


/***************************************************************************/
/***************************************************************************/
/***                                                                     ***/
/***                         GRAPHE                                      ***/
/***                                                                     ***/
/***************************************************************************/
/***************************************************************************/
/*

c3AMG::c3AMG(c3AMGS * aSym,double aRec) :
   mSym (aSym),
   mRec (aRec)
{
}

const double & c3AMG::Rec() const { return mRec; }
*/

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
