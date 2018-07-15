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

#include "NewOri.h"
//extern bool ERupnik_MM();

class cAppliFictObs : public cCommonMartiniAppli
{
    public:
        cAppliFictObs(int argc,char **argv);

    private:
        std::string mDir;
        std::string mPattern;
        std::string mOut;
};

cAppliFictObs::cAppliFictObs(int argc,char **argv) :
    mOut("_FObs")
{

    ElInitArgMain
    (
        argc, argv,
        LArgMain() << EAMC(mPattern,"Pattern of images"),
        LArgMain() << EAM (mOut,"Out",true,"Output file name")
    );
   #if (ELISE_windows)
        replace( mPattern.begin(), mPattern.end(), '\\', '/' );
   #endif

    SplitDirAndFile(mDir,mPattern,mPattern);
    
    cElemAppliSetFile anEASF(mPattern);
    const std::vector<std::string> * aSetName =   anEASF.SetIm();
    
    cNewO_NameManager *  aNM = NM(mDir);
    std::string aNameLTriplets = aNM->NameTopoTriplet(true);
    
    cXml_TopoTriplet  aLT = StdGetFromSI(aNameLTriplets,Xml_TopoTriplet);
    
    //pour chaque triplet recouper son elipse3d et genere les obs fict
    for (auto a3 : aLT.Triplets())
    {
        std::string  aName3R = aNM->NameOriOptimTriplet(true,a3.Name1(),a3.Name2(),a3.Name3());
        cXml_Ori3ImInit aXml3Ori = StdGetFromSI(aName3R,Xml_Ori3ImInit);
    
    }

    /*

   mNM = new cNewO_NameManager(mExtName,mPrefHom,mQuick,DirOfFile(mNameIm1),mNameOriCalib,"dat");


   // Structure d'image specialisee martini
   mIm1 = new cNewO_OneIm(*mNM,mNameIm1);
   mIm2 = new cNewO_OneIm(*mNM,mNameIm2);

 
       std::vector<cNewO_OneIm *>

       NOMerge_AddAllCams(mMergeStr,mVI);
       mMergeStr.DoExport();

       StdPack(const tMergeLPackH * aMPack,bool PondInvNorm,double aPdsSingle)
       
       ElPackHomologue aNM->PackOfName(const std::string & aN1,const std::string & aN2) const;



     */
    
   //cGenGaus3D aGG1(anEl);
   //std::vector<Pt3dr> aVP;

   //aGG1.GetDistribGaus(aVP,1+NRrandom3(2),2+NRrandom3(2),3+NRrandom3(2));




/* *********************************************************

    // Enregistre les points triples
    std::vector<Pt2df> aVP1Exp,aVP2Exp,aVP3Exp;
    std::vector<U_INT1> aVNb;
    for (tListM::const_iterator itM=aLM.begin() ; itM!=aLM.end() ; itM++)
    {
          if ((*itM)->NbSom()==3 )
          {
              aVP1Exp.push_back((*itM)->GetVal(0));
              aVP2Exp.push_back((*itM)->GetVal(1));
              aVP3Exp.push_back((*itM)->GetVal(2));
              aVNb.push_back((*itM)->NbArc());
          }
    }

    int aNb = (int)aVP1Exp.size();
    if (aNb<TNbMinTriplet)
    {
       aMap.Delete();
       return;
    }
    // Sauvegarde les triplet si assez

    mNM->WriteTriplet(aName3,aVP1Exp,aVP2Exp,aVP3Exp,aVNb);






*/
/* 
    map <img, triplet>
    pour chaque triplet
        recupere ses images/cams
        generate pts 3D
        reproject to pts 2D

        save to pack all cams
            {if file exists add otherwise create}
        

    inspiration pour save homol dans OriTiepRed/cOriCameraTiepRed.cpp"


void cCameraTiepRed::SaveHom(cCameraTiepRed* aCam2,const std::list<int> & aLBox)
{

    std::pair<CamStenope*,CamStenope*>  aPC((CamStenope *)NULL, (CamStenope *)NULL);
    if (mAppli.VerifNM())// (this != aCam2)
    {
       aPC = mAppli.NM().CamOriRel(NameIm(),aCam2->NameIm());
    }
    CamStenope* aCS1 = aPC.first;
    CamStenope* aCS2 = aPC.second;

    ElPackHomologue aRes;
    for (std::list<int>::const_iterator itI=aLBox.begin(); itI!=aLBox.end() ; itI++)
    {
         std::string aName = mAppli.NameHomol(NameIm(),aCam2->NameIm(),*itI);

         if (ELISE_fp::exist_file(aName))
         {
             ElPackHomologue aPack = ElPackHomologue::FromFile(aName);
             aRes.Add(aPack);

             // Verif
             if (aCS2)
             {
                 std::vector<double> aVD;
                 for (ElPackHomologue::const_iterator itP=aPack.begin(); itP!=aPack.end(); itP++)
                 {
                     double aDist;
                     aCS1->PseudoInterPixPrec(itP->P1(),*aCS2,itP->P2(),aDist);
                     aVD.push_back(aDist);
                 }
                 if (aVD.size())
                     std::cout << "Verif   CamOriRel " << MedianeSup(aVD) << "\n";
             }

         }
    }

    if (aRes.size())
    {
         std::string aKeyH = "NKS-Assoc-CplIm2Hom@"+ mAppli.StrOut() + "@dat";
         std::string aNameH = mAppli.ICNM()->Assoc1To2(aKeyH,NameIm(),aCam2->NameIm(),true);
         aRes.StdPutInFile(aNameH);
         // std::string aNameH = mAppli
    }

*/

}


int CPP_FictiveObsFin_main(int argc,char ** argv)
{
    cAppliFictObs AppliFO(argc,argv);

    return EXIT_SUCCESS;
 
}

