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

class cLinkTripl;
class cNOSolIn_AttrSom;
class cNOSolIn_AttrASym;
class cNOSolIn_AttrArc;
class cAppli_NewSolGolInit;
class cNOSolIn_Triplet;

typedef  ElSom<cNOSolIn_AttrSom,cNOSolIn_AttrArc>         tSomNSI;
typedef  ElArc<cNOSolIn_AttrSom,cNOSolIn_AttrArc>         tArcNSI;
typedef  ElSomIterator<cNOSolIn_AttrSom,cNOSolIn_AttrArc> tItSNSI;
typedef  ElArcIterator<cNOSolIn_AttrSom,cNOSolIn_AttrArc> tItANSI;
typedef  ElGraphe<cNOSolIn_AttrSom,cNOSolIn_AttrArc>      tGrNSI;
typedef  ElSubGraphe<cNOSolIn_AttrSom,cNOSolIn_AttrArc>   tSubGrNSI;


class cLinkTripl
{
     public :
         cLinkTripl(cNOSolIn_Triplet * aTrip,int aK1,int aK2,int aK3) :
            m3   (aTrip),
            mK1  (aK1),
            mK2  (aK2),
            mK3  (aK3)
         {
         }

         cNOSolIn_Triplet  *  m3;
         U_INT1               mK1;
         U_INT1               mK2;
         U_INT1               mK3;
         tSomNSI *            S1() const;
         tSomNSI *            S2() const;
         tSomNSI *            S3() const;
};



class cNOSolIn_AttrSom
{
     public :
         cNOSolIn_AttrSom(const std::string & aName,cAppli_NewSolGolInit & anAppli);
         cNOSolIn_AttrSom() :
             mCurRot (ElRotation3D::Id),
             mTestRot (ElRotation3D::Id)
         {}

         void AddTriplet(cNOSolIn_Triplet *,int aK0,int aK1,int aK2);
         // std::vector<cNOSolIn_Triplet *> & V3() {return mV3;}
         cNewO_OneIm * Im() {return mIm;}
         void ReInit();
         ElRotation3D & CurRot() {return mCurRot;}
         ElRotation3D & TestRot() {return mTestRot;}
     private :
         std::string                      mName;
         cAppli_NewSolGolInit *           mAppli;
         cNewO_OneIm *                    mIm;
         std::vector<cLinkTripl >         mLnk3;
         double                           mCurCostMin;
         ElRotation3D                     mCurRot;
         ElRotation3D                     mTestRot;
};


class cNOSolIn_AttrASym
{
     public :
         void AddTriplet(cNOSolIn_Triplet * aTrip,int aK1,int aK2,int aK3);
         std::vector<cLinkTripl> & Lnk3() {return mLnk3;}
     private :
         std::vector<cLinkTripl> mLnk3;
           
};
class cNOSolIn_AttrArc
{
     public :
           cNOSolIn_AttrArc(cNOSolIn_AttrASym *);
           cNOSolIn_AttrASym * ASym() {return mASym;}
     private :
           cNOSolIn_AttrASym * mASym;
};


class cNOSolIn_Triplet
{
      public :
          cNOSolIn_Triplet(tSomNSI * aS1,tSomNSI * aS2,tSomNSI *aS3,const cXml_Ori3ImInit &);
          void SetArc(int aK,tArcNSI *);
          tSomNSI * KSom(int aK) const {return mSoms[aK];}
          tArcNSI * KArc(int aK) const {return mArcs[aK];}

          void InitRot3Som();
          const ElRotation3D & RotOfSom(tSomNSI * aS)
          {
                if (aS==mSoms[0]) return ElRotation3D::Id;
                if (aS==mSoms[1]) return mR2on1;
                if (aS==mSoms[2]) return mR3on1;
                ELISE_ASSERT(false," RotOfSom");
                return ElRotation3D::Id;
          }

          


      private :
          tSomNSI *     mSoms[3];
          tArcNSI *     mArcs[3];
          ElRotation3D  mR2on1;
          ElRotation3D  mR3on1;
   // Gere les triplets qui vont etre desactives
          bool          mAlive;
};


class cAppli_NewSolGolInit
{
    public :
        cAppli_NewSolGolInit(int , char **);
        cNewO_NameManager & NM() {return *mNM;}

    private :
        void FinishNeighTriplet();
 
        void TestInitRot(tArcNSI * anArc,const cLinkTripl & aLnk);
        void TestOneTriplet(cNOSolIn_Triplet *);
        void SetNeighTriplet(cNOSolIn_Triplet *);

        void SetCurNeigh3(tSomNSI *);
        void SetCurNeigh2(tSomNSI *);

        void                 CreateArc(tSomNSI *,tSomNSI *,cNOSolIn_Triplet *,int aK0,int aK1,int aK2);
        std::string          mFullPat;
        std::string          mOriCalib;
        cElemAppliSetFile    mEASF;
        cNewO_NameManager  * mNM;
        bool                 mQuick;
        bool                 mTest;
 
        tGrNSI               mGr;
        std::map<std::string,tSomNSI *> mMapS;

// Variables temporaires pour charger un triplet 
        std::vector<tSomNSI *>  mVCur3;  // Tripelt courrant
        std::vector<tSomNSI *>  mVCur2;  // Adjcent au triplet courant
        int                     mFlag3;
        int                     mFlag2;
        cNOSolIn_Triplet *      mTestTrip;
        int                     mNbSom;
        int                     mNbArc;
        int                     mNbTrip;
        
};

/***************************************************************************/
/*                                                                         */
/*                 cNOSolIn_AttrSom                                        */
/*                                                                         */
/***************************************************************************/

cNOSolIn_AttrSom::cNOSolIn_AttrSom(const std::string & aName,cAppli_NewSolGolInit & anAppli) :
   mName        (aName),
   mAppli       (&anAppli),
   mIm          (new cNewO_OneIm(mAppli->NM(),mName)),
   mCurRot      (ElRotation3D::Id),
   mTestRot      (ElRotation3D::Id)
{
   ReInit();
}

void cNOSolIn_AttrSom::ReInit()
{
    mCurCostMin = 1e20;
}

void cNOSolIn_AttrSom::AddTriplet(cNOSolIn_Triplet * aTrip,int aK1,int aK2,int aK3)
{
    mLnk3.push_back(cLinkTripl(aTrip,aK1,aK2,aK3));
}


/***************************************************************************/
/*                                                                         */
/*                 cNOSolIn_Triplet                                        */
/*                                                                         */
/***************************************************************************/

cNOSolIn_Triplet::cNOSolIn_Triplet(tSomNSI * aS1,tSomNSI * aS2,tSomNSI *aS3,const cXml_Ori3ImInit & aTrip) :
    mR2on1 (Xml2El(aTrip.Ori2On1())),
    mR3on1 (Xml2El(aTrip.Ori3On1())),
    mAlive (true)
{
   mSoms[0] = aS1;
   mSoms[1] = aS2;
   mSoms[2] = aS3;
}

void cNOSolIn_Triplet::SetArc(int aK,tArcNSI * anArc)
{
   mArcs[aK] = anArc;
}

void cNOSolIn_Triplet::InitRot3Som()
{
     mSoms[0]->attr().CurRot() = ElRotation3D::Id;
     mSoms[1]->attr().CurRot() = mR2on1;
     mSoms[2]->attr().CurRot() = mR3on1;
}



/***************************************************************************/
/*                                                                         */
/*                 cNOSolIn_AttrASym                                       */
/*                                                                         */
/***************************************************************************/


void  cNOSolIn_AttrASym::AddTriplet(cNOSolIn_Triplet * aTrip,int aK1,int aK2,int aK3)
{
    mLnk3.push_back(cLinkTripl(aTrip,aK1,aK2,aK3));
}

tSomNSI * cLinkTripl::S1() const {return  m3->KSom(mK1);}
tSomNSI * cLinkTripl::S2() const {return  m3->KSom(mK2);}
tSomNSI * cLinkTripl::S3() const {return  m3->KSom(mK3);}

/***************************************************************************/
/*                                                                         */
/*                 cNOSolIn_AttrArc                                        */
/*                                                                         */
/***************************************************************************/

cNOSolIn_AttrArc::cNOSolIn_AttrArc(cNOSolIn_AttrASym * anASym) :
   mASym (anASym)
{
}

/***************************************************************************/
/*                                                                         */
/*                 cAppli_NewSolGolInit                                    */
/*                                                                         */
/***************************************************************************/

void cAppli_NewSolGolInit::SetCurNeigh3(tSomNSI * aSom)
{
     if (! aSom->flag_kth(mFlag3))
     {
        mVCur3.push_back(aSom);
        aSom->flag_set_kth_true(mFlag3);
     }
}

void cAppli_NewSolGolInit::SetCurNeigh2(tSomNSI * aSom)
{
     if (! aSom->flag_kth(mFlag2))
     {
        mVCur2.push_back(aSom);
        aSom->flag_set_kth_true(mFlag2);
     }
}

/*

    mTestTrip
    90 91 92     / 91 92 86

*/

void cAppli_NewSolGolInit::TestInitRot(tArcNSI * anArc,const cLinkTripl & aLnk)
{
      // Mp = M' coordonnees monde du triplet
      // M = coordonnees mondes courrament construite
      // La tranfo M' -> peut etre construite de deux maniere
      ElRotation3D aR1Mp2M  = aLnk.S1()->attr().CurRot().inv() * aLnk.m3->RotOfSom(aLnk.S1());
      ElRotation3D aR2Mp2M  = aLnk.S2()->attr().CurRot().inv() * aLnk.m3->RotOfSom(aLnk.S2());


      // ElRotation3D aTest = aR1Mp2M * aR2Mp2M.inv();
      // ElMatrix<double> aMT = aTest.Mat() -  ElMatrix<double>(3,true);
      ElMatrix<double>  aMT = aR1Mp2M.Mat() - aR2Mp2M.Mat();
      std::cout << "DIST MAT " << aMT.L2() << "\n";

      if (mTest)
      {
          ELISE_ASSERT(aLnk.S1()->attr().Im()->Name() == aLnk.m3->KSom(aLnk.mK1)->attr().Im()->Name(),"AAAAAaaaa");
          ELISE_ASSERT(aLnk.S2()->attr().Im()->Name() == aLnk.m3->KSom(aLnk.mK2)->attr().Im()->Name(),"AAAAAaaaa");

/*
          std::cout << "PERM " << (int) aLnk.mK1 << " " << (int) aLnk.mK2 << " " << (int) aLnk.mK3 << "\n";
          std::cout << mTestTrip->KSom(0)->attr().Im()->Name() << " "
                    << mTestTrip->KSom(1)->attr().Im()->Name() << " "
                    << mTestTrip->KSom(2)->attr().Im()->Name() << "\n";
          std::cout <<  aLnk.S1()->attr().Im()->Name() << " " <<  aLnk.m3->KSom(aLnk.mK1)->attr().Im()->Name() << "\n";
          std::cout <<  aLnk.S2()->attr().Im()->Name() << " " <<  aLnk.m3->KSom(aLnk.mK2)->attr().Im()->Name() << "\n";
          std::cout <<  aLnk.S3()->attr().Im()->Name() << " " <<  aLnk.m3->KSom(aLnk.mK3)->attr().Im()->Name() << "\n";
*/
          std::cout <<  anArc->s1().attr().Im()->Name() << " " <<  anArc->s2().attr().Im()->Name() << "\n";
          std::cout <<  aLnk.S3()->attr().Im()->Name()  << " " <<  aLnk.m3  << "\n";
          getchar();
      }
     // :Pt3dr 
}

void cAppli_NewSolGolInit::SetNeighTriplet(cNOSolIn_Triplet * aTripl)
{
    // On ajoute le triplet lui meme
    for (int aK=0 ; aK< 3 ; aK++)
    {
        tSomNSI * aKS = aTripl->KSom(aK);
        SetCurNeigh3(aKS);
        SetCurNeigh2(aKS);
    }
    aTripl->InitRot3Som();


    //  On recheche les sommet voisin 
    for (int aKA=0 ; aKA< 3 ; aKA++)
    {
         tArcNSI *  anA = aTripl->KArc(aKA);
         if (mTest) std::cout << "================ ARC ===== " << anA->s1().attr().Im()->Name() << " " <<  anA->s2().attr().Im()->Name() << "\n";

         std::vector<cLinkTripl> &  aLK3 = anA->attr().ASym()->Lnk3() ;
         for (int aK3=0 ; aK3 <int(aLK3.size()) ; aK3++)
         {
             tSomNSI * aSom = aLK3[aK3].S3();
             if (! aSom->flag_kth(mFlag3))
             {
                 if (! aSom->flag_kth(mFlag2))
                 {
                     SetCurNeigh2(aSom);
                 }
                 TestInitRot(anA,aLK3[aK3]);
                 // TestInitRot(
             }
         }
    }
}


void cAppli_NewSolGolInit::FinishNeighTriplet()
{
    for (int aK3=0 ; aK3<int(mVCur3.size()) ; aK3++)
    {
        mVCur3[aK3]->flag_set_kth_false(mFlag3);
    }
    for (int aK2=0 ; aK2<int(mVCur2.size()) ; aK2++)
    {
        mVCur2[aK2]->flag_set_kth_false(mFlag2);
        mVCur2[aK2]->attr().ReInit();
    }
}


void   cAppli_NewSolGolInit::CreateArc(tSomNSI * aS1,tSomNSI * aS2,cNOSolIn_Triplet * aTripl,int aK1,int aK2,int aK3)
{
     tArcNSI * anArc = mGr.arc_s1s2(*aS1,*aS2);
     if (anArc==0)
     {
         cNOSolIn_AttrASym * anAttrSym = new cNOSolIn_AttrASym;
         cNOSolIn_AttrArc anAttr12(anAttrSym);
         cNOSolIn_AttrArc anAttr21(anAttrSym);
         anArc = &(mGr.add_arc(*aS1,*aS2,anAttr12,anAttr21));
         mNbArc ++;
     }
     anArc->attr().ASym()->AddTriplet(aTripl,aK1,aK2,aK3);
     aTripl->SetArc(aK3,anArc);

     // return anArc;
}


cAppli_NewSolGolInit::cAppli_NewSolGolInit(int argc, char ** argv) :
    mQuick      (true),
    mTest       (true),
    mFlag3      (mGr.alloc_flag_som()),
    mFlag2      (mGr.alloc_flag_som()),
    mTestTrip   (0),
    mNbSom      (0),
    mNbArc      (0),
    mNbTrip     (0)
{
   std::string aNameT1;
   std::string aNameT2;
   std::string aNameT3;

   ElInitArgMain
   (
        argc,argv,
        LArgMain() << EAMC(mFullPat,"Pattern"),
        LArgMain() << EAM(mOriCalib,"OriCalib",true,"Orientation for calibration ", eSAM_IsExistDirOri)
                   << EAM(mQuick,"Quick",true,"Quick version",eSAM_IsBool)
                   << EAM(mTest,"Test",true,"Test for tuning",eSAM_IsBool)
                   << EAM(aNameT1,"Test1",true,"Name of first test image",eSAM_IsBool)
                   << EAM(aNameT2,"Test2",true,"Name of second test image",eSAM_IsBool)
                   << EAM(aNameT3,"Test3",true,"Name of third test image",eSAM_IsBool)
   );


   mEASF.Init(mFullPat);
   mNM = new cNewO_NameManager(mQuick,mEASF.mDir,mOriCalib,"dat");
   const cInterfChantierNameManipulateur::tSet * aVIm = mEASF.SetIm();

   for (int aKIm=0 ; aKIm <int(aVIm->size()) ; aKIm++)
   {
       const std::string & aName = (*aVIm)[aKIm];
       tSomNSI & aSom = mGr.new_som(cNOSolIn_AttrSom(aName,*this));
       mMapS[aName] = & aSom;
       mNbSom++;
       if (mTest)
       {
           ElMatrix<double> aR =  ElMatrix<double>::Rotation(aKIm,aKIm*10,aKIm*100);
           aSom.attr().TestRot() = ElRotation3D(Pt3dr(0,0,0),aR,true);
       }
   }


    cXml_TopoTriplet aXml3 =  StdGetFromSI(mNM->NameTopoTriplet(true),Xml_TopoTriplet);

    for
    (
         std::list<cXml_OneTriplet>::const_iterator it3=aXml3.Triplets().begin() ;
         it3 !=aXml3.Triplets().end() ;
         it3++
    )
    {
            tSomNSI * aS1 = mMapS[it3->Name1()];
            tSomNSI * aS2 = mMapS[it3->Name2()];
            tSomNSI * aS3 = mMapS[it3->Name3()];

ELISE_ASSERT(it3->Name1()<it3->Name2(),"AAAAAAAAAAAAaUyTR\n");
ELISE_ASSERT(it3->Name2()<it3->Name3(),"AAAAAAAAAAAAa0578\n");

if ((it3->Name1()=="IMGP9586.PEF") && (it3->Name2()=="IMGP9591.PEF") && (it3->Name3()=="IMGP9592.PEF"))
{
    std::cout << "KkkkkkkkkkkkkkkkkKKKkkkkkkkkkkkkkkkkkkkkkkkkkkKkkkkkkkkkkkkkkkkkkkkkkkk\n";
}



            if (aS1 && aS2 && aS3)
            {
                 mNbTrip++;

                 std::string  aN3 = mNM->NameOriGenTriplet
                                    (
                                        mQuick,
                                        true,  // ModeBin
                                        aS1->attr().Im(),
                                        aS2->attr().Im(),
                                        aS3->attr().Im()
                                    );
                 cXml_Ori3ImInit aXml3Ori = StdGetFromSI(aN3,Xml_Ori3ImInit);

                 if (mTest)
                 {
                     ElRotation3D aR1 = aS1->attr().TestRot();
                     ElRotation3D aR2 = aS2->attr().TestRot();
                     ElRotation3D aR3 = aS3->attr().TestRot();
                     ElRotation3D aR2On1 =  aR1.inv() * aR2;
                     ElRotation3D aR3On1 =  aR1.inv() * aR1;
                     aXml3Ori.Ori2On1() =  El2Xml(aR2On1);
                     aXml3Ori.Ori3On1() =  El2Xml(aR3On1);
                 }


                 cNOSolIn_Triplet * aTriplet = new cNOSolIn_Triplet(aS1,aS2,aS3,aXml3Ori);
/*
                 aS1->attr().AddTriplet(aTriplet,1,2,0);
                 aS2->attr().AddTriplet(aTriplet,0,2,1);
                 aS3->attr().AddTriplet(aTriplet,0,1,2);
*/
                 CreateArc(aS1,aS2,aTriplet,0,1,2);
                 CreateArc(aS2,aS3,aTriplet,1,2,0);
                 CreateArc(aS3,aS1,aTriplet,2,0,1);

                 if ((it3->Name1()==aNameT1) && (it3->Name2()==aNameT2) && (it3->Name3()==aNameT3))
                 {
                     mTestTrip = aTriplet;
                 }
            }
    }

    if (mTestTrip)
    {
        std::cout << "mTestTrip " << mTestTrip << "\n";

        std::cout <<  "GLOB NbS = " <<  mNbSom 
                 << " NbA " << mNbArc  << ",Da=" <<   (2.0 *mNbArc)  / (mNbSom*mNbSom) 

                 << " Nb3 " << mNbTrip  << ",D3=" << (3.0 *mNbTrip)  / (mNbArc*mNbSom)  << "\n";

        // cAppli_NewSolGolInit::SetNeighTriplet
        SetNeighTriplet(mTestTrip);
        std::cout << "NbIn Neih " <<  mVCur2.size() << "\n";
        for (int aK=0 ; aK< int(mVCur2.size()) ; aK++)
        {
            std::cout << "  Neigh " << mVCur2[aK]->attr().Im()->Name() ;
            if (  mVCur2[aK]->flag_kth(mFlag3)) std::cout << " *** ";
            std::cout << "\n";
        }
    }
   
}


int CPP_NewSolGolInit_main(int argc, char ** argv)
{
    cAppli_NewSolGolInit anAppli(argc,argv);
    return EXIT_SUCCESS;
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
