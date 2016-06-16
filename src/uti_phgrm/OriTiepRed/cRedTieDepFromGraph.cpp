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

#include "OriTiepRed.h"
NS_OriTiePRed_BEGIN

void Banniere_Ratafia()
{
    std::cout << "***************************************************\n";
    std::cout << "*                                                 *\n";
    std::cout << "*       R-eduction                                *\n";
    std::cout << "*       A-utomatic of                             *\n"; 
    std::cout << "*       T-ie points                               *\n"; 
    std::cout << "*       A-iming at                                *\n"; 
    std::cout << "*       F-ast                                     *\n"; 
    std::cout << "*       I-mage                                    *\n"; 
    std::cout << "*       A-erotriangulation                        *\n"; 
    std::cout << "*                                                 *\n";
    std::cout << "***************************************************\n";
}

/**********************************************************************/
/*                                                                    */
/*                         cAttArcSymGrRedTP                          */
/*                                                                    */
/**********************************************************************/

cAttArcSymGrRedTP::cAttArcSymGrRedTP(const cXml_Ori2Im & anOri) :
    mOri (anOri)
{
}

const cXml_Ori2Im & cAttArcSymGrRedTP::Ori() const
{
   return mOri;
}

/**********************************************************************/
/*                                                                    */
/*                         cAttArcASymGrRedTP                         */
/*                                                                    */
/**********************************************************************/

cAttArcASymGrRedTP::cAttArcASymGrRedTP(cAttArcSymGrRedTP * aASym,bool direct) :
   mASym   (aASym),
   mDirect (direct)
{
}

const cXml_Ori2Im & cAttArcASymGrRedTP::Ori() const
{
   return mASym->Ori();
}

double & cAttArcASymGrRedTP::Recouv() {return mRecouv;}

const Box2dr & cAttArcASymGrRedTP::Box() const
{
   return mDirect ? mASym->Ori().Box1() : mASym->Ori().Box2();
}

const double & cAttArcASymGrRedTP::Foc() const
{
   return mDirect ? mASym->Ori().Foc1() : mASym->Ori().Foc2();
}

/**********************************************************************/
/*                                                                    */
/*                         cV2ParGRT                                  */
/*                                                                    */
/**********************************************************************/

void cV2ParGRT::AddSom(tSomGRTP * aSom)
{
   mVSom.push_back(aSom);
}

const std::vector<tSomGRTP *> & cV2ParGRT::VSom() const
{
   return mVSom;
}

/**********************************************************************/
/*                                                                    */
/*                         cAttSomGrRedTP                             */
/*                                                                    */
/**********************************************************************/

cAttSomGrRedTP::cAttSomGrRedTP(cAppliGrRedTieP & anAppli,const std::string & aName) :
   mAppli     (&anAppli),
   mCamGlob   (0),
   mName      (aName),
   mNbPtsMax  (0),
   mRecSelec  (0.0),
   mRecCur    (0.0),
   mMTD       (cMetaDataPhoto::CreateExiv2(mName)),
   mSzDec     (anAppli.SzPixDec()/mMTD.FocPix())
{
}


int & cAttSomGrRedTP::NbPtsMax() {return mNbPtsMax;}
const std::string & cAttSomGrRedTP::Name() const {return mName;}
double & cAttSomGrRedTP::RecSelec() {return mRecSelec;}
double & cAttSomGrRedTP::RecCur() {return mRecCur;}
Box2dr & cAttSomGrRedTP::BoxIm() {return mBoxIm;}
double  cAttSomGrRedTP::SzDec() const {return mSzDec;}

/**********************************************************************/
/*                                                                    */
/*                         cAppliGrRedTieP                            */
/*                                                                    */
/**********************************************************************/

void cAppliGrRedTieP::SetSelected(tSomGRTP * aSom)
{

    aSom->flag_set_kth_true(mFlagSel);
    aSom->flag_set_kth_true(mFlagCur);
    mPartParal.back()->AddSom(aSom);
    mFiloCur.pushlast(aSom);
    for (tIterArcGRTP  itA=aSom->begin(mSubAll) ; itA.go_on(); itA++)
    {
        tSomGRTP & aS2 = (*itA).s2();
        double aRec = (*itA).attr()->Recouv();
        ElSetMax(aS2.attr()->RecCur(),aRec);
        ElSetMax(aS2.attr()->RecSelec(),aRec);
    }

}



tSomGRTP * cAppliGrRedTieP::GetNextBestSom()
{
   mPcc.pcc (
                mFiloCur,
                mSubNone,
                mSubAll,   // pds => 1.0 par defaut, donc cela convient
                eModePCC_Somme
            );

   tSomGRTP * aRes = 0;

   for (int aKs=0 ; aKs<int(mVSom.size()) ; aKs++)
   {
       tSomGRTP * aSom = mVSom[aKs];
/*
       std::cout << " NNN=" << aSom->attr()->Name() 
                 << " P=" << mPcc.pds(*aSom)  
                 << " Rc=" << aSom->attr()->RecCur() 
                 << " S=" << aSom->flag_kth(mFlagSel)
                 << "\n";
*/
       if (! aSom->flag_kth(mFlagSel))
       {
           if (aRes==0)
           {
              aRes = aSom;
           }
           else
           {
              if (aSom->attr()->RecCur() < aRes->attr()->RecCur())
              {
                 aRes = aSom;
              }
              else if (    (aSom->attr()->RecCur() == aRes->attr()->RecCur())
                        && ( mPcc.pdsDef(*aSom) <  mPcc.pdsDef(*aRes))
                      )
              {
                 aRes = aSom;
              }
           }
       }
   }

   if ((aRes!=0)  && (aRes->attr()->RecCur() > mRecMax))
   {
      aRes = 0;
   }

   return aRes;
}



bool cAppliGrRedTieP::OneItereSelection()
{

     mFiloCur.clear();
     // Pour S0, on prend le non selec le + proche (recouvert) des deja selec
     tSomGRTP * aS0 = 0;
     for (int aKs=0 ; aKs<mNbSom ; aKs++)
     {
        tSomGRTP * aSom = mVSom[aKs];
        aSom->attr()->RecCur() = 0;
        if (
                (! aSom->flag_kth(mFlagSel))  
             && ((aS0==0)|| (aSom->attr()->RecSelec() > aS0->attr()->RecSelec()))
           )
        {
            aS0 = aSom;
        }
     }


     if (aS0==0) 
       return false;


    cV2ParGRT * aSetPart = new cV2ParGRT;
    mPartParal.push_back(aSetPart);
    SetSelected(aS0);

    // recherche de l'ensemble courant
    const std::vector<tSomGRTP *> & aVS = aSetPart->VSom();

    bool ContAdd = true;
    while (ContAdd)
    {
       if (mNbP <= int (aVS.size()  ))
       {
          ContAdd = false;
       }
       else
       {
          tSomGRTP * aSom =  GetNextBestSom();
          if (aSom)
          {
              SetSelected(aSom);
          }
          else
          {
             ContAdd = false;
          }
       }
    }


    // On efface la selection courante
    for (int aKs=0 ; aKs<int(aVS.size()) ; aKs++)
    {
        aVS[aKs]->flag_set_kth_false(mFlagCur);
    }
    
    //
    return true;
}

void cAppliGrRedTieP::Show()
{
    double aRecGr = 0.0;
    for (int aKP=0; aKP<int(mPartParal.size()) ; aKP++)
    {
        double aRecPart = 0.0;
        const std::vector<tSomGRTP *> & aVS = mPartParal[aKP]->VSom();
        if (aKP!=0) std::cout << "**********************************************************\n";
        
        for (int aKS1=0 ; aKS1<int(aVS.size()) ; aKS1++)
        {
            double aRec = 0.0;
            for (int aKS2=0 ; aKS2<int(aVS.size()) ; aKS2++)
            {
               tArcGRTP * anArc = mGr.arc_s1s2(*(aVS[aKS1]),(*aVS[aKS2]));

               if (anArc)
               {
                  aRec += anArc->attr()->Recouv();
               }
            }
            std::cout << "   " << aVS[aKS1]->attr()->Name() << " R=" << aRec << "\n";
            aRecPart += aRec;
        }
        aRecGr += aRecPart;
    }
    std::cout 
              << "NbCl " << mPartParal.size()
              << " Rec=" << aRecGr
              << "\n";
   getchar();
}

double  cAppliGrRedTieP::SzPixDec() const
{
   return mSzPixDec;
}


cAppliGrRedTieP::cAppliGrRedTieP(int argc,char ** argv) :
    mUseOR       (true),
    mQuick       (true),
    mNbP         (-1),
    mFlagSel     (mGr.alloc_flag_som()),
    mFlagCur     (mGr.alloc_flag_som()),
    mRecMax      (DefRecMaxModeIm),
    mShowPart    (false),
    mAppliTR     (0),
    mSzPixDec    (4000)
{
   // Read parameters 
   MMD_InitArgcArgv(argc,argv);
   ElInitArgMain
   (
         argc,argv,
         LArgMain()  << EAMC(mPatImage, "Pattern of images",  eSAM_IsPatFile),
         LArgMain()  << EAM(mCalib,"OriCalib",true,"Calibration folder if any")
                     << EAM(mUseOR,"UseOR",true,"Use Relative Orientation (Def =true)")
                     << EAM(mNbP,"NbP",true,"Nb Process, def use all")
                     << EAM(mRecMax,"RecMax",true,"Max overlap acceptable in two parallely processed images")
                     << EAM(mShowPart,"ShowP",true,"Show Partition (def=false)")
                     << EAM(mSzPixDec,"SzPixDec",true,"Sz of decoupe in pixel")
   );


   {
      std::string aComATR = /* MM3dBinFile("OriRedTieP")i*/ + " " + mPatImage + " FromRG=true";
      if (EAMIsInit(&mCalib))
      {
         aComATR = aComATR + " OriCalib=" + mCalib ;
      }
      std::vector<char *> * aVA = new std::vector<char *>(ToArgMain(aComATR));
      mAppliTR =  new cAppliTiepRed(aVA->size(),&((*aVA)[0]),true);
   }
  // cAppliTiepRed::cAppliT
 // std::vector<char *> ToArgMain(const std::string & aStr);


    cElemAppliSetFile::Init(mPatImage);
    mNoNM = cVirtInterf_NewO_NameManager::StdAlloc(mDir,mCalib,mQuick);

    if (!EAMIsInit(&mNbP))
    {
       mNbP = MMNbProc();
    }

    for (int aK=0 ; aK<int(mSetIm->size()) ; aK++)
    {
        const std::string & aName = (*mSetIm)[aK];
        tSomGRTP & aSom = mGr.new_som(new cAttSomGrRedTP(*this,aName));
        mDicoSom[aName] = &aSom;
        mVSom.push_back(&aSom);
    }
    mNbSom = mVSom.size();

    std::string aNameCple =  mUseOR                                ?
                             mNoNM->NameListeCpleOriented(true)    :
                             mNoNM->NameListeCpleConnected(true)   ;

    cSauvegardeNamedRel  aLCple = StdGetFromPCP(aNameCple,SauvegardeNamedRel);

    for (std::vector<cCpleString>::const_iterator itCp=aLCple.Cple().begin() ; itCp!=aLCple.Cple().end() ; itCp++)
    {
        tSomGRTP * aS1 = mDicoSom[itCp->N1()];
        tSomGRTP * aS2 = mDicoSom[itCp->N2()];

        if ((aS1!=0) && (aS2!=0))
        {
             cXml_Ori2Im anOri = mNoNM->GetOri2Im(itCp->N1(),itCp->N2());
             cAttArcSymGrRedTP * aAASym = new cAttArcSymGrRedTP(anOri);
             mGr.add_arc(*aS1,*aS2,new cAttArcASymGrRedTP(aAASym,true),new cAttArcASymGrRedTP(aAASym,false));
        }
    }
     

   // Calcul du nombre de connexion max

    for (tIterSomGRTP itS=mGr.begin(mSubAll);itS.go_on();itS++)
    {
        tSomGRTP & aS1 = (*itS);
        for (tIterArcGRTP  itA=aS1.begin(mSubAll) ; itA.go_on(); itA++)
        {
             const cXml_Ori2Im & anOri = (*itA).attr()->Ori();
             ElSetMax(aS1.attr()->NbPtsMax(),anOri.NbPts());
        }
    }

   // Calcul du taux de recouvrement de chaque arc
    for (tIterSomGRTP itS=mGr.begin(mSubAll);itS.go_on();itS++)
    {
        tSomGRTP & aS1 = (*itS);
        Pt2dr aPInf( 1e60, 1e60);
        Pt2dr aPSup(-1e60,-1e60);
        for (tIterArcGRTP  itA=aS1.begin(mSubAll) ; itA.go_on(); itA++)
        {
             tSomGRTP & aS2 = (*itA).s2();
             const cXml_Ori2Im & anOri = (*itA).attr()->Ori();
             (*itA).attr()->Recouv() =  anOri.NbPts() / double(aS2.attr()->NbPtsMax());
             Box2dr aBox = (*itA).attr()->Box();
             aPInf = Inf(aPInf,aBox._p0);
             aPSup = Sup(aPSup,aBox._p1);
        }
        aS1.attr()->BoxIm() = Box2dr(aPInf,aPSup);

        std::cout <<  aS1.attr()->Name() << " " <<  aS1.attr()->BoxIm().sz() << " " << aS1.attr()->SzDec() << "\n";
    }

    while(OneItereSelection());

    if (mShowPart)
    {
       Show();
    }
}

NS_OriTiePRed_END


NS_OriTiePRed_USE

int  Ratafia_Main(int argc,char ** argv)
{
    cAppliGrRedTieP anAppli(argc,argv);

    Banniere_Ratafia();
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
