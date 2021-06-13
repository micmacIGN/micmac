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
#include "SolInitNewOri.h"

bool BugNanNewOri = false;

/**********************************************************/
/*                                                        */
/*          Composantes connexes                          */
/*                                                        */
/**********************************************************/

void cAppli_NewSolGolInit::ResetFlagCC()
{

    for (int  aK3=0 ; aK3<int (mV3.size()) ; aK3++)
    {
         mV3[aK3]->Flag().set_kth_false(mFlag3CC);
    }
}


void cAppli_NewSolGolInit::NumeroteCC()
{
    int aNumCC = 0;
    // Parse all triplet
    for (int  aK3=0 ; aK3<int (mV3.size()) ; aK3++)
    {
        cNOSolIn_Triplet * aTri0 = mV3[aK3];

        // If the triplet has not been marked, it's a new component
        if ( !aTri0->Flag().kth(mFlag3CC))
        {
            // std::vector<cNOSolIn_Triplet*> * aCC = new std::vector<cNOSolIn_Triplet*>;
            // Create a new component
            cNO_CC_TripSom * aNewCC3S = new cNO_CC_TripSom;
            aNewCC3S->mNumCC = aNumCC;  // give it a number
            mVCC.push_back(aNewCC3S);   // add it to the vector of component
            std::vector<cNOSolIn_Triplet*> * aCC3 = &(aNewCC3S->mTri); // Quick acces to vec of tri in the CC
            std::vector<tSomNSI *> * aCCS = &(aNewCC3S->mSoms); // Quick accessto som

            // Calcul des triplets 
            aCC3->push_back(aTri0);  // Add triplet T0
            aTri0->Flag().set_kth_true(mFlag3CC);// Mark it as explored
            aTri0->NumCC() = aNumCC;  // Put  right num to T0
            int aKCur = 0;
            // Traditional loop of CC : while  no new inexplored neighboor
            while (aKCur!=int(aCC3->size())) 
            {
               cNOSolIn_Triplet * aTri1 = (*aCC3)[aKCur];
               // For each edge of the current triplet
               for (int aKA=0 ; aKA<3 ; aKA++)
               {
                  // Get triplet adjacent to this edge and parse them
                  std::vector<cLinkTripl> &  aLnk = aTri1->KArc(aKA)->attr().ASym()->Lnk3();
                  for (int aKL=0 ; aKL<int(aLnk.size()) ; aKL++)
                  {
                     // If not marked, mark it and push it in aCC3, return it was added
                     if (SetFlagAdd(*aCC3,aLnk[aKL].m3,mFlag3CC))
                     {
                          aLnk[aKL].m3->NumCC() = aNumCC;
                     }
                  }
               }
               aKCur++;
            }

            // Compute the summit of the CC, it's easy, just be carrful to get them only once
            int aFlagSom = mGr.alloc_flag_som();
            for (int aKT=0 ; aKT<int(aCC3->size()) ; aKT++)
            {
                cNOSolIn_Triplet * aTri = (*aCC3)[aKT];
                for (int aKS=0 ;  aKS<3 ; aKS++)
                {
                    SetFlagAdd(*aCCS,aTri->KSom(aKS),aFlagSom);
                }
            }
            FreeAllFlag(*aCCS,aFlagSom);
            mGr.free_flag_som(aFlagSom);

            // std::cout << "NbTriii " << aCC3->size() << " NbSooom " << aCCS->size() << "\n";
            aNumCC++;
        }
    }
    FreeAllFlag(mV3,mFlag3CC);
    // ResetFlagCC();
    // std::cout << "NUMMMCCCC " <<  aNumCC << "\n";
}

/**********************************************************/
/*                                                        */
/*          Orientation                                   */
/*                                                        */
/**********************************************************/


/*
    AJoute tous les voisins dans la file d'attente (ou modifie leur priorite)
*/

void  cAppli_NewSolGolInit::AddSOrCur(tSomNSI * aSom,const ElRotation3D & aR)
{

    // std::cout << "ADDSOOm " << aSom->attr().Im()->Name() << "\n";

    SetFlagAdd(mVSOrCur,aSom,mFlagSOrCur);
    // ELISE_ASSERT(Added,"cAppli_NewSolGolInit::AddSOrCur");
    aSom->attr().CurRot() = aR;

    int aFlagMaj = mGr.alloc_flag_som();
    std::vector<tSomNSI *> aVMaj;

    for (tItANSI anItA=aSom->begin(mSubAll) ; anItA.go_on(); anItA++)
    {
         tSomNSI * aS2 = & (*anItA).s2();
         if  (aS2->flag_kth(mFlagSOrCur))
         {
              // std::cout << "  S2 " << aS2->attr().Im()->Name() << "\n";
              std::vector<cLinkTripl> & aVLnK = (*anItA).attr().ASym()->Lnk3() ;
              for (int aKL=0 ; aKL<int(aVLnK.size()) ; aKL++)
              {
                  tSomNSI * aS3 = aVLnK[aKL].S3();
                  cNOSolIn_Triplet * aTri = aVLnK[aKL].m3;
                  if ((! aS3->flag_kth(mFlagSOrCur)) && (! aS3->flag_kth(mFlagSOrGerm)))
                  {
                      if (SetFlagAdd(mVSOrCdt,aS3,mFlagSOrCdt))
                      {
                          aS3->attr().ResetGainByTriplet() ;
                          mHeapSom.push(aS3);
                      }
                      aS3->attr().AddGainByTriplet(aTri->GainArc()) ;
                      // aS3->attr().GainByTriplet() += 1;
                      SetFlagAdd(aVMaj,aS3,aFlagMaj);
                      // std::cout << "    S3  " << aS3->attr().Im()->Name()  << " G=" << aTri->GainArc() << "\n";
                  }
              }
         }
    }

    for (int aK=0 ; aK<int(aVMaj.size()) ; aK++)
    {
        mHeapSom.MAJ(aVMaj[aK]);
    }

    FreeSet(aVMaj,aFlagMaj);

}


/*
     C1toTri, C2toTri , C3toTri
     C1toCur, C2toCur , C3toCur

      TriTocur * CKtoTri   =  CKToCur
      TriTocur = CKToCur  * CKtoTri-1

*/

bool SomSpec(cNOSolIn_Triplet * aTri)
{
return     (aTri->KSom(0)->attr().Im()->Name()=="OIS-Reech_Vol10_1986_FR3989_0267_013.tif")
       &&  (aTri->KSom(1)->attr().Im()->Name()=="OIS-Reech_Vol11_1986_FR3989_0297_015.tif")
       &&  (aTri->KSom(2)->attr().Im()->Name()=="OIS-Reech_Vol12_1986_FR3989_0323_015.tif");
}

double cAppli_NewSolGolInit::ReMoyOneTriplet(cNOSolIn_Triplet * aTri)
{
     ElMatrix<double> aMTri2Cur(3,3,0.0);
     ElMatrix<double> anId(3,true);
     std::vector<ElMatrix<double> >  aVM; aVM.reserve(3);
     std::vector<Pt3dr> aVCTri;
     std::vector<Pt3dr> aVCCur;

     ///calcul de la rotation du passage (local->global)
     for (int aKS=0 ; aKS<3 ; aKS++)
     {
         tSomNSI * aSom =  aTri->KSom(aKS);
         const ElRotation3D &   aRCur =  aSom->attr().CurRot();
         const ElRotation3D &   aRTri =  aTri->RotOfSom(aSom);

         ElMatrix<double> aMKTri2Cur = aRCur.Mat() * aRTri.Mat().transpose();
         aVM.push_back(aMKTri2Cur);
         aMTri2Cur =  aMTri2Cur + aMKTri2Cur;
         aVCTri.push_back(aRTri.tr());
         aVCCur.push_back(aRCur.tr());
     }
     aMTri2Cur = NearestRotation(aMTri2Cur * (1.0/3.0));

     double aSomDCur=0;
     double aSomDTri=0;

     ///som de translations entre des images voisin dans repere local (aSomDTri) et globale (aSomDCur) 
     for (int aKS=0 ; aKS<3 ; aKS++)
     {
          aSomDCur += euclid(aVCCur[aKS]-aVCCur[(aKS+1)%3]);
          aSomDTri += euclid(aVCTri[aKS]-aVCTri[(aKS+1)%3]);
     }
     ///facteur d'echelle entre le modele local et global
     double aLambda = aSomDCur / aSomDTri;




     ///calcul de la translation => aOffsTr + aMTri2Cur * PTri * aLambda = PCur
     Pt3dr aOffsTr(0,0,0);
     for (int aKS=0 ; aKS<3 ; aKS++)
     {
          aOffsTr =  aOffsTr + aVCCur[aKS] - aMTri2Cur*aVCTri[aKS] * aLambda;
     }
     aOffsTr = aOffsTr / 3.0;

     if (BugNanNewOri) 
        std::cout << "OOOoaOffsTr= " << aOffsTr << "\n";

     double aSomDistMat = 0; 
     double aSomDistTr = 0; 
     for (int aKS=0 ; aKS<3 ; aKS++)
     {
          aSomDistMat += aMTri2Cur.L2(aVM[aKS]);
          aSomDistTr += euclid(aVCCur[aKS]-aOffsTr - aMTri2Cur*aVCTri[aKS] * aLambda);
     }
     aSomDistTr /=  (aLambda*2);
     aSomDistTr  *= aTri->BOnH();

     aSomDistMat = sqrt(aSomDistMat/2.0);

     if (BugNanNewOri) 
        std::cout << "aSomDistMataSomDistMat= " << aSomDistMat << "\n";

     double aSomDist =  aSomDistTr + aSomDistMat;
     double anEcart = ElMax(mLastPdsMedRemoy,mCoherMed12);
     if (aSomDist < (10*anEcart))
     {
        double aPds = 1 / (1+ElSquare(aSomDist/(2*anEcart)));

        for (int aKS=0 ; aKS<3 ; aKS++)
        {
     //  aTr + aMTri2Cur * PTri * aLambda = PCur
             tSomNSI * aSom =  aTri->KSom(aKS);

             Pt3dr  aTrK = aOffsTr + aMTri2Cur*aVCTri[aKS] * aLambda;
             ElMatrix<double> aMK = aMTri2Cur * aTri->RotOfSom(aSom).Mat();

             if (aSom->attr().CamInOri())
             {
                 const ElRotation3D &   aRCur =  aSom->attr().CurRot();

                 aTrK = aRCur.tr();
                 aMK = aRCur.Mat();
             }

            // ElMatrix<double> aMKTri2Cur = aRCur.Mat() * aRTri.Mat().transpose();

             //repartitin d'erreur; les valuers global change ici
             aSom->attr().SomPdsReMoy() += aPds;
             aSom->attr().SomTrReMoy () = aSom->attr().SomTrReMoy () + aTrK * aPds;
             aSom->attr().SomMatReMoy() = aSom->attr().SomMatReMoy() + aMK * aPds;


             Pt3dr aPMed = aOffsTr + aMTri2Cur* aTri->PMed() * aLambda;
             aSom->attr().SomPMedReM () = aSom->attr().SomPMedReM () + aPMed * aPds;
        }
     }

if (SomSpec(aTri))
{
     std::cout << "LLLLambda= " << aLambda  << " Nb3 " << aTri->Nb3() << "\n";
     std::cout <<  aTri->KSom(0)->attr().Im()->Name() << " " 
               <<  aTri->KSom(1)->attr().Im()->Name() << " " 
               <<  aTri->KSom(2)->attr().Im()->Name() << " " 
               << "\n";
     std::cout << "SD " << aSomDist << " " << anEcart << "\n";
     // if (BugNanNewOri) std::cout << "LLLLambda= " << aLambda << "\n";
     // getchar();
}

     return aSomDist;
}

void cAppli_NewSolGolInit::StatTrans(Pt3dr & aMoy,double & aDist)
{
    aMoy = Pt3dr(0,0,0);
    aDist = 0;
    int aNbS = (int)mVSOrCur.size();
    for (int aKS=0 ; aKS<aNbS ; aKS++)
    {
         tSomNSI * aSom = mVSOrCur[aKS];
         Pt3dr aTr = aSom->attr().CurRot().tr();
         aMoy = aMoy + aTr;
         aDist += square_euclid(aTr);
    }
    aMoy = aMoy / aNbS;
    aDist = sqrt(ElMax(0.0,aDist/aNbS - square_euclid(aMoy)));
}

void cAppli_NewSolGolInit::ReMoyByTriplet()
{
    Pt3dr aTr0,aTrFin;
    double aDist0,aDistFin;

    StatTrans(aTr0,aDist0);

    mLastEcartReMoy.clear();
    // For all oriented som reset the stat that will accumulate the different rotation
    for (int aKS=0 ; aKS <  int(mVSOrCur.size()) ; aKS++)
    {
        tSomNSI * aSom = mVSOrCur[aKS];
        aSom->attr().SomPdsReMoy() = 0;              // Weigthing
        aSom->attr().SomTrReMoy () = Pt3dr(0,0,0);   // Translation
        aSom->attr().SomPMedReM () = Pt3dr(0,0,0);   // P3D Med ?
        aSom->attr().SomMatReMoy() = ElMatrix<double>(3,3,0.0);  // Rotation
    }

    for (int aK3=0 ; aK3<int(mV3Use4Ori.size()) ; aK3++)
    {
static int aCptRMT=0 ;aCptRMT++;
BugNanNewOri = (aCptRMT==3683440);


        double aDist = ReMoyOneTriplet(mV3Use4Ori[aK3]);
if (std_isnan(aDist))
{
    std::cout << "aCptRMT== " << aCptRMT << " NanDist\n";
    getchar();
}
        mLastEcartReMoy.push_back(aDist);
    }

static int aCpt=0; aCpt++;
std::cout << " ENTER MED " << aCpt << "\n";  // Nan
    mLastPdsMedRemoy = MedianeSup(mLastEcartReMoy);
std::cout << "--------- END  MED \n";


    for (int aKS=0 ; aKS <  int(mVSOrCur.size()) ; aKS++)
    {
        tSomNSI * aSom = mVSOrCur[aKS];
        // A faire avant modif du poid
        aSom->attr().SomPMedReM() = aSom->attr().SomPMedReM() / aSom->attr().SomPdsReMoy();
        double aPdsThis = ElMax(0.01, aSom->attr().SomPdsReMoy() * 0.1);
        aSom->attr().SomPdsReMoy() += aPdsThis;
        aSom->attr().SomTrReMoy ()  = aSom->attr().SomTrReMoy ()  +  aSom->attr().CurRot().tr() * aPdsThis;
        aSom->attr().SomMatReMoy ()  = aSom->attr().SomMatReMoy ()  +  aSom->attr().CurRot().Mat() * aPdsThis;

        Pt3dr aTr =  aSom->attr().SomTrReMoy () / aSom->attr().SomPdsReMoy();
        ElMatrix<double> aMat = NearestRotation(aSom->attr().SomMatReMoy () * (1.0/aSom->attr().SomPdsReMoy()));

        // std::cout << "ppppPDS " << aSom->attr().SomPdsReMoy() << "\n";
        // aSom->attr().SomTrReMoy () = Pt3dr(0,0,0);
        // aSom->attr().SomMatReMoy() = ElMatrix<double>(3,3,0.0);
        // std::cout << "EUCLID " << euclid(aTr-aSom->attr().CurRot().tr()) << " " << sqrt(aMat.L2(aSom->attr().CurRot().Mat())) << "\n";
        if (mActiveRemoy)
           aSom->attr().CurRot() = ElRotation3D(aTr,aMat,true);
    }


    StatTrans(aTrFin,aDistFin);

    for (int aKS=0 ; aKS <  int(mVSOrCur.size()) ; aKS++)
    {
        tSomNSI * aSom = mVSOrCur[aKS];
        Pt3dr aTr = aSom->attr().CurRot().tr();
        aTr = (aTr-aTrFin)  * (aDist0/aDistFin) + aTr0;
        aSom->attr().CurRot().tr() = aTr;
    }
}

/*
   M coord monde (deja construit)  , T  coordonnes triplet

   R1.Cur() =   Cam1ToM   ;  R2.Cur() =   Cam2ToM
   m3.RofS(S1) = Cam1ToL  ;  m3.RofS(S2) = Cam2ToL ; m3.RofS(S3) = Cam3toL

   Cam3ToM =  LtoM *  Cam3ToL

   LtoM  =   Cam1ToM *    Cam1ToL-1
*/

class cNO_SolEstimRot
{
    public :
       cNO_SolEstimRot(const ElRotation3D & aRot,double aCost,cNOSolIn_Triplet * aTri) :
            mRot (aRot),
            mCost (aCost),
            mTri  (aTri)
       {
       }
       ElRotation3D       mRot;
       double             mCost;
       cNOSolIn_Triplet * mTri;
};

class cCmp_cNO_SolEstimRot
{
    public :
        bool operator () (const cNO_SolEstimRot & aS1,const cNO_SolEstimRot & aS2)
        {
             return aS1.mCost < aS2.mCost;
        }
};


double DistRotAbs(const ElRotation3D & aR1,const ElRotation3D & aR2,double aBOnH)
{
   return sqrt(aR1.Mat().L2(aR2.Mat())) + euclid(aR1.tr()-aR2.tr())  * aBOnH;
}

ElRotation3D cNOSolIn_AttrSom::EstimRot(tSomNSI * aSom)
{
     int aFlagOr = mAppli->FlagSOrCur();
     int aCpt=0;

     std::vector<cNO_SolEstimRot> aVS;
     std::vector<double> aVProf;
     
     // Calcul pour chaque triplet de la rotation initiale

     for (int aKL=0 ; aKL<int(mLnk3.size()) ; aKL++)
     {
         cLinkTripl & aLnk = mLnk3[aKL];
         tSomNSI * aS1 = aLnk.S1();
         tSomNSI * aS2 = aLnk.S2();
         bool isInit = aS1->flag_kth(aFlagOr) && aS2->flag_kth(aFlagOr) ;
         if (isInit)
         {
             cNOSolIn_Triplet * aTri = aLnk.m3;
             SetFlagAdd(mAppli->V3Use4Ori(),aTri,mAppli->Flag3UsedForOri());
             const ElRotation3D & aC1ToM = aS1->attr().CurRot();
             const ElRotation3D & aC2ToM = aS2->attr().CurRot();

             const ElRotation3D & aC1ToL = aTri->RotOfSom(aS1);
             const ElRotation3D & aC2ToL = aTri->RotOfSom(aS2);
             const ElRotation3D & aC3ToL = aTri->RotOfSom(aSom);

             ElMatrix<double> aMatLtoM1 = aC1ToM.Mat() * aC1ToL.Mat().transpose();
             ElMatrix<double> aMatLtoM2 = aC2ToM.Mat() * aC2ToL.Mat().transpose();
             ElMatrix<double> aMatLtoM = NearestRotation((aMatLtoM1+aMatLtoM2)*0.5);

          //   (Lambda,TR , aMatLtoM ) * aC1ToL  ~ aC1ToM
          //   (Lambda,TR , aMatLtoM ) * aC2ToL  ~ aC2ToM
          //   (Lambda,TR , aMatLtoM ) * aC3ToL  ~ aC3ToM
          //   TR + Lambda * aMatLtoM * aC1ToL.tr = aC1ToM.tr

             Pt3dr aC1MTri = aMatLtoM * aC1ToL.tr();
             Pt3dr aC2MTri = aMatLtoM * aC2ToL.tr();
             Pt3dr aV12Tri = aC2MTri - aC1MTri;

             Pt3dr aC1MIni   = aC1ToM.tr();
             Pt3dr aC2MIni   = aC2ToM.tr();
             Pt3dr aV12Ini   = aC2MIni - aC1MIni;
             double aLambda = euclid(aV12Ini) /  ElMax(euclid(aV12Tri),1e-20);

             Pt3dr aTR1 =  aC1MIni -  aC1MTri* aLambda;
             Pt3dr aTR2 =  aC2MIni -  aC2MTri* aLambda;
             Pt3dr aTR = (aTR1+aTR2) /2.0;


             Pt3dr aTr3M = aTR + aMatLtoM * aC3ToL.tr() * aLambda;


             ElRotation3D  aRC3toM = ElRotation3D(aTr3M,aMatLtoM*aC3ToL.Mat() ,true) ;

             //aV12Tri = aV12Tri * aLambda;


             double aD = sqrt(aMatLtoM.L2(aMatLtoM2)) + euclid(aV12Tri*aLambda-aV12Ini) * aTri->BOnH() ;
             mAppli->VDistEstimRot().push_back(aD);
             double aProf = euclid(aTri->PMed()-aC3ToL.tr()) * aLambda;
             aVProf.push_back(aProf);

             // std::cout << "MEEDD " << aTri->PMed()  << " Pr " << aProf << " L " << aLambda << "\n";
             // std::cout << " DDD  " <<   aD << " Med " << aD/mAppli->CoherMed12()    << " B/H " << aTri->BOnH() << "\n";


             aVS.push_back(cNO_SolEstimRot(aRC3toM,aD,aTri));


             aCpt++;
         }
     }

    // std::cout << "MMEDDDS " << MedianeSup(mAppli->VDistEstimRot()) << " " << mAppli->CoherMed12() << "\n";

     cCmp_cNO_SolEstimRot aCmp;
     std::sort(aVS.begin(),aVS.end(),aCmp);
     double aProf = MedianeSup(aVProf);


     // Recherche d'un noyau, egal a un des NbInitEvalRot  plus coherent 
     int aNbInit = ElMin(NbInitEvalRot,int(aVS.size()));

     double aDistMin = 1e30;
     int aKMin = -1;
     for (int aK0=0 ; aK0<aNbInit ; aK0++)
     {
          double aSomD = 0.0;
          const ElRotation3D & aR0 = aVS[aK0].mRot;
          for (int aK1 = 0 ; aK1<int(aVS.size()) ; aK1++)
          {
               if (aK1!=aK0)
               {
                    const ElRotation3D & aR1 = aVS[aK1].mRot;
                    // double aDist = sqrt(aR0.Mat().L2(aR1.Mat())) + euclid(aR0.tr()-aR1.tr()) / aProf;
                    double aDist = DistRotAbs(aR0,aR1,1.0/aProf);  // sqrt(aR0.Mat().L2(aR1.Mat())) + euclid(aR0.tr()-aR1.tr()) / aProf;
                    aSomD += CoutAttenueTetaMax(aDist,2*  mAppli->CoherMed12());
               }
          }
          if (aSomD<aDistMin)
          {
              aDistMin = aSomD;
              aKMin = aK0;
          }
     }

     ElRotation3D aRes = aVS[aKMin].mRot;
     if (mAppli->IterLocEstimRot())
     {
         int aNbIter = 4;
         for (int aKIter=0 ; aKIter<aNbIter ; aKIter++)
         {
              Pt3dr aTr (0,0,0);
              ElMatrix<double> aSomM(3,3,0.0);
              double aSomPds=0.0;
              for (int aKR = 0 ; aKR<int(aVS.size()) ; aKR++)
              {
                  const ElRotation3D & aR  = aVS[aKR].mRot;
                  double aDist = sqrt(aRes.Mat().L2(aR.Mat())) + euclid(aRes.tr()-aR.tr()) / aProf;
                  // if (aKIter==(aNbIter-1)) std::cout << "DD = " << aDist << "\n";
                  if (aDist < 6 * mAppli->CoherMed12())
                  {
                      double aPds = 1 /(1 + ElSquare(aDist/(2*mAppli->CoherMed12())));
                      aSomPds += aPds;
                      aTr = aTr + aR.tr() * aPds;
                      aSomM = aSomM + aR.Mat() * aPds;

                  }
              }
              if (aSomPds >0)
              {
                 aRes = ElRotation3D(aTr/aSomPds,NearestRotation(aSomM*(1/aSomPds)),true);
              }
              // std::cout << "SOOM PDS " << aSomPds/aVS.size()  << "\n";
         }
      }

     



     for (int aK1=0 ; aK1<int(aVS.size()) ; aK1++)
     {
        for (int aK2=aK1+1 ; aK2<int(aVS.size()) ; aK2++)
        {
              ElRotation3D aR1 = aVS[aK1].mRot;
              ElRotation3D aR2 = aVS[aK2].mRot;
              // std::cout << " Mat " << sqrt(aR1.Mat().L2(aR2.Mat())) << " " << euclid(aR1.tr()-aR2.tr()) << "\n";
        }
     }


     // std::cout << "HSomm " << Im()->Name()  << " G=" << aSom->attr().CalcGainByTriplet() << " NbL=" << aCpt  << " on " << mLnk3.size() << "\n";

     // getchar();

     return aRes;
}


void  cAppli_NewSolGolInit::FreeSet(std::vector<tSomNSI*>  & aV,int aFlag)
{
    FreeAllFlag(aV,aFlag);
    mGr.free_flag_som(aFlag);
    aV.clear();
}

void cAppli_NewSolGolInit::FreeTriplet(std::vector<cNOSolIn_Triplet*>  & aV,int aFlag)
{
    FreeAllFlag(aV,aFlag);
    mAllocFlag3.flag_free(aFlag);
    aV.clear();

}


tSomNSI * cAppli_NewSolGolInit::GetBestSom()
{
    tSomNSI * aSom;
    if (mHeapSom.pop(aSom))
       return aSom;
    return 0;
}



void cAppli_NewSolGolInit::CalculOrient(cNOSolIn_Triplet * aGerm)
{
    // Alloc a certain number of flag to mark submit
    mFlagSOrCur = mGr.alloc_flag_som();
    mFlagSOrCdt = mGr.alloc_flag_som();
    mFlagSOrGerm = mGr.alloc_flag_som();
    mFlag3UsedForOri = mAllocFlag3.flag_alloc(); // Flag for triplet


    SetFlagAdd(mV3Use4Ori,aGerm,mFlag3UsedForOri); // Add seed and mark it

    if (mHasInOri)  // Dont comment for now the branch with InOri
    {
         for (tItSNSI anItS=mGr.begin(mSubAll) ; anItS.go_on(); anItS++)
         // for (int aKS=0 ; aKS<3 ; aKS++)
         {
              tSomNSI & aSom = *anItS;
              // tSomNSI &  aSom = *(aGerm->KSom(aKS));
              CamStenope *   aCam = aSom.attr().CamInOri();
              if (aCam)
              {
                  AddSOrCur(&aSom,aCam->Orient().inv());
                  // AddSOrCur(&aSom,aCam->Orient().inv());
                  SetFlagAdd(mVSOrGerm,&aSom,mFlagSOrGerm);
                  // aV1.push_back(aGerm->RotOfK(aKS).tr());
                  // aV2.push_back(aCam->Orient().inv().tr());
              }
         }
    }
    else
    {
        // Put the 3 sommit in the heap
        for (int aKS=0 ; aKS<3 ; aKS++)
        {
             AddSOrCur(aGerm->KSom(aKS),aGerm->RotOfK(aKS));
             SetFlagAdd(mVSOrGerm,aGerm->KSom(aKS),mFlagSOrGerm);
        }
    }

    tSomNSI * aSom;
    int aCpt = 0;
    while ((aSom = GetBestSom()))
    {
         ElRotation3D aRot = aSom->attr().EstimRot(aSom);
         AddSOrCur(aSom,aRot);
         for (int aK=0 ; aK<3 ; aK++)
             ReMoyByTriplet();

         aCpt++;
         if ((aCpt % 20)==0)
            std::cout << "          CalculOrient, done " << aCpt << "soms , T=" << mChrono.uval() << "\n";
    }

    for (int aK=0 ; aK<mNbIterLast ; aK++)
        ReMoyByTriplet();



    double aEc80 = KthValProp(mLastEcartReMoy,0.8);
    std::cout << "ReMoy  Med=" << mLastPdsMedRemoy  << "  80%=" << aEc80  << " Ec80/Med12 " << aEc80/mCoherMed12 << "\n";

    FreeSet(mVSOrCur,mFlagSOrCur);
    FreeSet(mVSOrCdt,mFlagSOrCdt);
    FreeSet(mVSOrGerm,mFlagSOrGerm);
    FreeTriplet(mV3Use4Ori,mFlag3UsedForOri);
    mVDistEstimRot.clear();

    if (0)
    {
         for (tItSNSI anItS=mGr.begin(mSubAll) ; anItS.go_on(); anItS++)
         {
             tSomNSI * aS1 = &(*anItS);
             std::cout << "AftOr " << aS1->attr().Im()->Name() 
                       << " " <<  aS1->attr().Lnk3().size()
                       << " " <<  aS1->attr().NbGainByTriplet()
                       << "\n";

         }
    }

    
}



void  cAppli_NewSolGolInit::CalculOrient(cNO_CC_TripSom * aCC)
{
     std::cout << "    CC of CalculOrient , Nb Som " << aCC->mSoms.size() << "\n";
     cNOSolIn_Triplet * aGerm0 =0;
     double aBesCoherCost = 1e30;

     for (int aK=0 ; aK< int(aCC->mTri.size()) ; aK++)
     {
         cNOSolIn_Triplet * aTri = aCC->mTri[aK];
         if ((!mHasInOri) || (aTri->TripletIsInOri()))
         {
             if (aTri->CostArc()<aBesCoherCost)
             {
                 aBesCoherCost = aTri->CostArc();
                 aGerm0 = aTri;
             }
         }
     }
     ELISE_ASSERT(aGerm0!=0,"Cannot compute germ in CalculOrient (due to InOri ?)");

     CalculOrient(aGerm0);
}


void  cAppli_NewSolGolInit::CalculOrient()
{
    for (int aKC=0 ;  aKC<int(mVCC.size()) ; aKC++)
       CalculOrient(mVCC[aKC]);
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
