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
#include "MultTieP.h"

bool  FileModeBin(const std::string & aName)
{
   std::string aPost = StdPostfix(aName);
   if (aPost=="dat")
     return true;
   if (aPost=="txt")
     return false;
   ELISE_ASSERT(false,"FileModeBin");
   return false;
}

class cSetIntDyn
{
    public :
          cSetIntDyn(const  std::vector<int> &);
          bool operator < (const cSetIntDyn &);
    private :
          std::vector<int> mVI;
};


cSetIntDyn::cSetIntDyn(const  std::vector<int> & aVI) :
    mVI (aVI)
{
     std::sort(mVI.begin(),mVI.end());
}

bool cSetIntDyn::operator < (const cSetIntDyn & aSID)
{
    size_t aNb = mVI.size();
    if (aNb <  aSID.mVI.size()) 
       return true;
    if (aNb >  aSID.mVI.size()) 
       return false;
    for (size_t aK=0 ; aK<aNb ; aK++)
    {
        if ( mVI[aK] <  aSID.mVI[aK])
           return true;
        if ( mVI[aK] >  aSID.mVI[aK])
           return false;
    }
    return false;
}


/*******************************************************************/
/*                                                                 */
/*           Homologue -> cVarSizeMergeTieP                        */
/*                                                                 */
/*******************************************************************/


typedef cVarSizeMergeTieP<Pt2df,cCMT_NoVal>  tMergeRat;
typedef cStructMergeTieP<tMergeRat>  tMergeStrRat;
typedef  std::map<std::string,int>  tDicNumIm;

template <class Type> std::vector<int> VecIofVecIT(const std::vector<cPairIntType<Type> >  & aVecIT)
{
   std::vector<int> aRes;
   for (int aK=0 ; aK<int(aVecIT.size()) ; aK++)
       aRes.push_back(aVecIT[aK].mNum);

   return aRes;
}
std::vector<Pt2dr> VecPtofVecIT(const std::vector<cPairIntType<Pt2df> >  & aVecIT)
{
   std::vector<Pt2dr> aRes;
   for (int aK=0 ; aK<int(aVecIT.size()) ; aK++)
       aRes.push_back(ToPt2dr(aVecIT[aK].mVal));

   return aRes;
}





// Conserve les numeros initiaux

const std::list<tMergeRat *> &  CreatePMul
                                (
                                    cVirtInterf_NewO_NameManager * aVNM,
                                    const std::vector<std::string> * aVIm
                                )
{
    
    tDicNumIm aDicoNumIm;
    std::map<std::string,CamStenope *> aDicCam;
    for (int aKIm=0 ; aKIm<int(aVIm->size()); aKIm++)
    {
       const std::string & aNameIm = (*aVIm)[aKIm];
       aDicoNumIm[aNameIm] = aKIm;
       aDicCam[aNameIm] = aVNM->CalibrationCamera((*aVIm)[aKIm]);
    }

    std::string aNameCple = aVNM->NameListeCpleConnected(true);
    cSauvegardeNamedRel aLCple = StdGetFromPCP(aNameCple,SauvegardeNamedRel);

    tMergeStrRat & aMergeStruct =  *(new tMergeStrRat(aVIm->size(),false));

    for 
    (
        std::vector<cCpleString>::iterator itV=aLCple.Cple().begin();
        itV != aLCple.Cple().end();
        itV++
    )
    {
        tDicNumIm::iterator it1 = aDicoNumIm.find(itV->N1());
        tDicNumIm::iterator it2 = aDicoNumIm.find(itV->N2());
        if ((it1!=aDicoNumIm.end()) && (it2!=aDicoNumIm.end()))
        {
            CamStenope * aCS1 = aDicCam[itV->N1()] ;
            CamStenope * aCS2 = aDicCam[itV->N2()] ;

            std::vector<Pt2df> aVP1,aVP2;
            aVNM->LoadHomFloats(itV->N1(),itV->N2(),&aVP1,&aVP2);
            int aNum1 = it1->second;
            int aNum2 = it2->second;

            for (int aKP = 0  ; aKP<int(aVP1.size()) ; aKP++)
            {
                Pt2dr aP1 = aCS1->L3toF2(Pt3dr(aVP1[aKP].x,aVP1[aKP].y,1.0));
                Pt2dr aP2 = aCS2->L3toF2(Pt3dr(aVP2[aKP].x,aVP2[aKP].y,1.0));
                aMergeStruct.AddArc(ToPt2df(aP1),aNum1,ToPt2df(aP2),aNum2,cCMT_NoVal());
            }
        }
 
    }
    aMergeStruct.DoExport();
    return  aMergeStruct.ListMerged();
}

/*******************************************************************/
/*                                                                 */
/*                Conversion                                       */
/*                                                                 */
/*******************************************************************/

class cAppliConvertToNewFormatHom
{
    public :
        cAppliConvertToNewFormatHom(int argc,char ** argv);
    private :
        std::string         mPatImage;
        std::string         mDest;
        cElemAppliSetFile   mEASF;
        const std::vector<std::string> * mFilesIm;
        bool                             mDoNewOri;
        cVirtInterf_NewO_NameManager *   mVNM;
        bool                             mModeBin;
};


cAppliConvertToNewFormatHom::cAppliConvertToNewFormatHom(int argc,char ** argv) :
      mDoNewOri (true),
      mModeBin  (false)
{
    
   ElInitArgMain
   (
         argc,argv,
         LArgMain()  << EAMC(mPatImage, "Pattern of images",  eSAM_IsPatFile)
                     << EAMC(mDest, "Output"),
         LArgMain()  << EAM(mDoNewOri,"DoNewOri",true,"Tuning")
                     << EAM(mModeBin,"Bin",true,"Binary format")
   );

   mEASF.Init(mPatImage);
   mFilesIm = mEASF.SetIm();

   if (mDoNewOri)
   {
        std::string aCom =  MM3dBinFile("TestLib NO_AllOri2Im ") + QUOTE(mPatImage) + " GenOri=false ";
        System(aCom);

        std::cout << "DONE NO_AllOri2Im \n";
        // mm3d TestLib NO_AllOri2Im IMGP70.*JPG  GenOri=false 
   }

   mVNM = cVirtInterf_NewO_NameManager::StdAlloc(mEASF.mDir,"",true);
   // Conserve les numeros initiaux des images
   const std::list<tMergeRat *> &  aLMR = CreatePMul  (mVNM,mFilesIm);
   std::cout << "DONE PMUL " << aLMR.size() << " \n";

   cSetTiePMul * aSetOutPM = new cSetTiePMul();
   aSetOutPM->SetCurIms(*mFilesIm);

   for (std::list<tMergeRat *>::const_iterator itMR=aLMR.begin() ; itMR!=aLMR.end() ; itMR++)
   {
       std::vector<int> aVI = VecIofVecIT((*itMR)->VecIT());
       std::vector<Pt2dr> aVPts = VecPtofVecIT((*itMR)->VecIT());
       cSetPMul1ConfigTPM * aConf = aSetOutPM->OneConfigFromVI(aVI);
       aConf->Add(aVPts,(*itMR)->NbArc());
   }


   aSetOutPM->Save(mDest);

   // cSetTiePMul * aSetInPM = new cSetTiePMul();
   // aSetInPM->AddFile(aName);
}


int ConvertToNewFormatHom_Main(int argc,char ** argv)
{
    cAppliConvertToNewFormatHom(argc,argv);
    return EXIT_SUCCESS;
    // :
}

int UnionFiltragePHom_Main(int argc,char ** argv)
{
   std::string aPatHom,aPatIm,aDest;
   ElInitArgMain
   (
         argc,argv,
         LArgMain()  << EAMC(aPatHom, "Pattern of HomFile",  eSAM_IsPatFile)
                     << EAMC(aDest,"Destination"),
         LArgMain()  
                     << EAM(aPatIm,"Filter",true,"Filter for selecting images")
   );
    
   cElemAppliSetFile   anEASF_Hom(aPatHom);
   cSetTiePMul * aSetOutPM = new cSetTiePMul();

   if (EAMIsInit(&aPatIm))
   {
        cElemAppliSetFile   anEASF_Im(aPatIm);
        aSetOutPM->SetFilter(*(anEASF_Im.SetIm()));
   }

   const std::vector<std::string> * aVFileH = anEASF_Hom.SetIm();

   for (int aKH=0 ; aKH<int(aVFileH->size())  ; aKH++)
   {
       aSetOutPM->AddFile(anEASF_Hom.mDir+(*aVFileH)[aKH]);
   }

   aSetOutPM->Save(aDest);
   return EXIT_SUCCESS;
}

/*********************************************************************/
/*                                                                   */
/*                  cCelImTPM                                        */
/*                                                                   */
/*********************************************************************/

cCelImTPM::cCelImTPM(const std::string & aNameIm,int anId) :
   mNameIm (aNameIm),
   mId     (anId)
{
}


/*********************************************************************/
/*                                                                   */
/*                  cDicoImTPM                                       */
/*                                                                   */
/*********************************************************************/

cCelImTPM * cDicoImTPM::AddIm(const std::string & aNameIm,bool & IsNew)
{
   cCelImTPM * aRes = mName2Im[aNameIm];
   IsNew = (aRes==0);
   if (IsNew)
   {
       aRes = new cCelImTPM(aNameIm,mNum2Im.size());
       mName2Im[aNameIm] = aRes;
       mNum2Im.push_back(aRes);
   }
   return aRes;
}


/*********************************************************************/
/*                                                                   */
/*                  cSetPMul1ConfigTPM                               */
/*                                                                   */
/*********************************************************************/


cSetPMul1ConfigTPM::cSetPMul1ConfigTPM(const  std::vector<int> & aVIdIm,int aNbPts) :
    mVIdIm (aVIdIm),
    mNbIm  (mVIdIm.size()),
    mNbPts (aNbPts),
    mVXY   (),
    mPrec  (1/500.0)
{
   mVXY.reserve(2*mNbIm*mNbPts);
}

void cSetPMul1ConfigTPM::Add(const std::vector<Pt2dr> & aVP,int aNbArc)
{
    ELISE_ASSERT(mNbIm==int(aVP.size()),"cSetPMul1ConfigTPM::Add NbPts");
    for (int aKP=0 ; aKP<mNbIm ; aKP++)
    {
         mVXY.push_back(Double2Int(aVP[aKP].x));
         mVXY.push_back(Double2Int(aVP[aKP].y));
    }
    mNbPts++;
    mVNbA.push_back(aNbArc);
}

int cSetPMul1ConfigTPM::NbArc(int aKP) const
{
   return mVNbA[aKP];
}

/*********************************************************************/
/*                                                                   */
/*                  cSetTiePMul                                      */
/*                                                                   */
/*********************************************************************/

cSetTiePMul::cSetTiePMul() :
    mSetFilter (0)
{
}

void cSetTiePMul::SetFilter(const std::vector<std::string> & aVIm )
{
    delete mSetFilter;
    mSetFilter = new std::set<std::string>(aVIm.begin(),aVIm.end());
}



void cSetTiePMul::SetCurIms(const std::vector<std::string> & aVIm) 
{
    mNumConvCur.clear();
    for (std::vector<std::string>::const_iterator itN=aVIm.begin() ; itN!=aVIm.end() ; itN++)
    {
        if ((mSetFilter==0) ||  (BoolFind(*mSetFilter,*itN)))
        {
           bool IsNew;
           cCelImTPM * aCel = AddIm(*itN,IsNew);
           mNumConvCur.push_back(aCel->mId);
        }
        else
        {
           mNumConvCur.push_back(-1);
        }
    }
}

cSetPMul1ConfigTPM * cSetTiePMul::OneConfigFromVI(const std::vector<INT> & aVI)
{
    cSetPMul1ConfigTPM * aRes = mMapConf[aVI];
    if (aRes==0)
    {
       aRes = new cSetPMul1ConfigTPM(aVI,0);
       mMapConf[aVI] = aRes;
       mPMul.push_back(aRes);
    }
    return aRes;
}

cCelImTPM * cSetTiePMul::AddIm(const std::string & aNameIm,bool & IsNew)
{
   return mDicoIm.AddIm(aNameIm,IsNew);
}


static const std::string HeaderBeginTPM="BeginHeader-MicMacTiePointFormat";
static const std::string HeaderEndTPM="EndHeader";

void cSetTiePMul::Save(const std::string & aName)
{
    ELISE_fp::MkDirRec(DirOfFile(aName));
    ELISE_fp aFp(aName.c_str(),ELISE_fp::WRITE,false, FileModeBin(aName) ? ELISE_fp::eBinTjs : ELISE_fp::eTxtTjs);

    aFp.SetFormatDouble("%.3lf");
        


    aFp.write_line(HeaderBeginTPM);
    aFp.write_line("   Version=0.0.0");
    aFp.write_line(HeaderEndTPM);

    aFp.PutCommentaire("=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=");
    aFp.write_U_INT4(mDicoIm.mNum2Im.size());
    aFp.PutCommentaire("Nb Images");
    for (int aK=0 ; aK<int(mDicoIm.mNum2Im.size()) ; aK++)
    {
        cCelImTPM * aCel = mDicoIm.mNum2Im[aK];
// std::cout << aCel->mNameIm << "\n";
        aFp.write_line(aCel->mNameIm +"="+ToString(aCel->mId));
        // aFp.PutLine();
        // aFp.write_U_INT4(aCel->mId);
        // aFp.PutLine();
    }
    aFp.PutCommentaire("=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=");
    aFp.write_U_INT4(mPMul.size());
    aFp.PutCommentaire("Number of configuration");
    aFp.PutLine();
    aFp.PutCommentaire("NbPts this config,  NbIm in this config");
    aFp.PutCommentaire("Im0 Im1 Im2..");
    aFp.PutCommentaire("x(0,0) y(0,0) ... x(0,NbIm-1) y(0,NbIm-1) NbEgdes(0)");
    aFp.PutCommentaire("x(1,0) y(1,0) ... x(1,NbIm-1) y(1,NbIm-1) NbEgdes(1)");
    aFp.PutCommentaire(".....");
    aFp.PutCommentaire("x(NbPts-1,0) y(NbPts-1,0) ... x(NbPts-1,NbIm-1) y(NbPts-1,NbIm-1) NbEgdes(NbPts-1)");
    aFp.PutLine();

    for (int aKConf=0 ; aKConf<int(mPMul.size()) ; aKConf++)
    {
         cSetPMul1ConfigTPM * aConf = mPMul[aKConf];

         aFp.write_U_INT4(aConf->mNbPts);
         aFp.write_U_INT4(aConf->mNbIm);
         aFp.PutLine();
          
         for (int aKIm=0 ; aKIm<aConf->mNbIm ; aKIm++)
         {
               aFp.write_U_INT4(aConf->mVIdIm[aKIm]);
         }
         aFp.PutLine();

         for (int aKP=0 ; aKP<aConf->mNbPts ; aKP++)
         {
             for (int aKIm=0 ; aKIm<aConf->mNbIm ; aKIm++)
             {
                Pt2dr aPt = aConf->Pt(aKP,aKIm);
                aFp.write_REAL8(aPt.x);
                aFp.write_REAL8(aPt.y);
             }
             aFp.write_INT4(aConf->NbArc(aKP));
             aFp.PutLine();
         }

         aFp.PutCommentaire("====");
    }

    aFp.close();
}

void cSetTiePMul::AddFile(const std::string & aName)
{
    ELISE_fp aFp(aName.c_str(),ELISE_fp::READ,false, FileModeBin(aName) ? ELISE_fp::eBinTjs : ELISE_fp::eTxtTjs);


    std::string aHeader = aFp.std_fgets();
    ELISE_ASSERT(aHeader==HeaderBeginTPM,"Bad Header in MicMac Multiple Point Format");

    // std::cout << "HHHHhhH=["<< aHeader << "]\n";
    bool Cont = true;
    while (Cont)
    {
       std::string aS= aFp.std_fgets();
       if (aS== HeaderEndTPM)
       {
          Cont = false;
       }
    }
    
    int aNbIm = aFp.read_U_INT4();
    std::vector<std::string> aVNameIm;
    aFp.PasserCommentaire();
    for (int aK=0 ; aK<aNbIm ; aK++)
    {
       std::string aS= aFp.std_fgets();
       std::string aName,anId;
       SplitIn2ArroundEq(aS,aName,anId);
       aVNameIm.push_back(aName);
       int aVerifK ;
       FromString(aVerifK,anId);
       ELISE_ASSERT(aK==aVerifK,"Bad Image Id expectation in cSetTiePMul::AddFile");
    }
    
    SetCurIms(aVNameIm);

    int aNbConfig = aFp.read_U_INT4();

    std::cout << "NB CONFIG=" <<  aNbConfig << "\n";
    for (int aKConf=0 ; aKConf < aNbConfig ; aKConf++)
    {
        int aNbPt = aFp.read_U_INT4();
        int aNbIm   = aFp.read_U_INT4();
        // std::cout << "PT " << aNbPt << " Immm " << aNbIm << "\n";
        std::vector<int> aVIm;
        std::vector<int> aVImOk;
        for (int aKIm=0 ; aKIm<aNbIm ; aKIm++)
        {
            int aNum =  aFp.read_U_INT4();
            aNum = mNumConvCur.at(aNum);
            aVIm.push_back(aNum);
            if (aNum>=0)
               aVImOk.push_back(aNum);
        }
        cSetPMul1ConfigTPM * aConfig = (aVImOk.size()>=2) ? OneConfigFromVI(aVImOk) : 0;
        
        for (int aKPt = 0 ; aKPt<aNbPt ; aKPt++)
        {
            std::vector<Pt2dr> aVPt;
            for (int aKIm=0 ; aKIm<aNbIm ; aKIm++)
            {
                 double aX = aFp.read_REAL8();
                 double aY = aFp.read_REAL8();
                 if (aVIm[aKIm] >= 0)
                 {
                    aVPt.push_back(Pt2dr(aX,aY));
                 }
            }
            int aNbArc = aFp.read_U_INT4(); // NbEdges
            if (aConfig)
               aConfig->Add(aVPt,aNbArc);
            // std::cout << "NbbArrrc " << aNbArc << "\n";
            // getchar();
        }
    }

    aFp.close();

    Save("DUP.txt");
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
