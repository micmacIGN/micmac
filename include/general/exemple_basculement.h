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

#ifndef  _EXEMPLE_BASCULEMENT_H_
#define  _EXEMPLE_BASCULEMENT_H_


// D'autre classe de photogrammetrie, dont, entre autre le basculement

class cSolBasculeRig;
class cRansacBasculementRigide;
class cL2EqObsBascult;


/*****************************************************************************/
/*                                                                           */
/*     Classes pour resoudre le "baculement". Sont inconnues                 */
/*   une Rot-Trans+ un facteur d'echelle                                     */
/*                                                                           */
/*****************************************************************************/

extern bool DEBUG_GCP_BASCULE;

class cSolBasculeRig : public cTransfo3D
{
    public :
       cSolBasculeRig
       (
           const Pt3dr & aPAv,
           const Pt3dr & aPApres,
           const ElMatrix<double> & aRot,  //  =>  mMatR
           double aLambda                  // mL
       );
        virtual std::vector<Pt3dr> Src2Cibl(const std::vector<Pt3dr> &) const ;

       static cSolBasculeRig SBRFromElems( const Pt3dr & aTr,const ElMatrix<double> & aRot,const double & aLambda );
       cSolBasculeRig Inv() const;
       // VR1 et VR2 sont des orientation de Camera, sens Cam-> Monde, sans outlier
       // renvoie une solution type LSQ M2->M1
       static cSolBasculeRig SolM2ToM1(const std::vector<ElRotation3D> & aVR1, const std::vector<ElRotation3D> & aVR2);


       // Renvoie une solution au sens des moindre L2, initialisee par RANSAC
       static cSolBasculeRig StdSolFromPts
                             (
                                   const std::vector<Pt3dr> & aV1,
                                   const std::vector<Pt3dr> & aV2,
                                   const std::vector<double> * aVPds=0, // si 0 ts les pds valent 1
                                   int   aNbRansac             = 200,
                                   int   aNbL2                 = 5
                             );


       Pt3dr operator()(const Pt3dr &) const;  // mTr + mMatR * aP * mL
       static cSolBasculeRig  Id();

       const Pt3dr & Tr() const ;
       const ElMatrix<double> & Rot() const;
       double           Lambda() const;

        // Si B=cSolBasculeRig est telle que  P2=B(P1) permet de
        // passer d'un systeme de coordonnee S1   a s2 alors
        // TransformC2M donne transforme des orientation externes (exprimee
        // dans le sens Cam-> Monde) de S1 a S2
        // Assez elementaire.
        
       ElRotation3D TransformOriC2M(const ElRotation3D &) const;

       void QualitySol(  const std::vector<ElRotation3D> & aVR1 ,
                         const std::vector<ElRotation3D> & aVR2,
                         double & aDMatr,
                         double & aDCentr
            );

    private :

        ElMatrix<double> mMatR;
        double           mL;
        Pt3dr            mTr;
};

/* => Redondant avec TransformOriC2M 
cSolBasculeRig  BascFromVRot
                (
                     const std::vector<ElRotation3D> & aVR1 ,
                     const std::vector<ElRotation3D> & aVR2,
                     std::vector<Pt3dr> &              aVP1,
                     std::vector<Pt3dr> &              aVP2
                );
*/


cSolBasculeRig SolElemBascRigid
               (
                    const Pt3dr & aAvant1, const Pt3dr & aAvant2, const Pt3dr & aAvant3,
                    const Pt3dr & aApres1, const Pt3dr & aApres2, const Pt3dr & aApres3
               );



class cRansacBasculementRigide
{
    public :
        cRansacBasculementRigide(bool WithSpeed);
        ~cRansacBasculementRigide();


        const cSolBasculeRig & BestSol() const;
        void ExploreAllRansac(int aNbRanSac=1000000000) ;


        void AddExemple(const Pt3dr & aAvant,const Pt3dr & aApres,const Pt3dr * aSpeedApres,const std::string & aName);

        // Clos et estim la position centrale par Barrycentre
        bool CloseWithTrGlob(bool Svp= false);
        // Clos et estim la position centrale sur l'exemple K
        bool CloseWithTrOnK(int aK,bool Svp= false);
        int  CurK() const;
        const std::vector<Pt3dr>  & PAvant() const;
        const  std::vector<Pt3dr> & PApres()  const;
        const  std::vector<std::string> & Names()  const;
        void EstimateDelay();
        double   Delay() const;

          bool SolIsInit() const;
         double EstimLambda() const;

    private :

          void AssertSolInit() const;

         void  TestNewSol(const cSolBasculeRig &) ;

         double CostSol(const cSolBasculeRig &) const;
         //  Solution en rendant aussi exact que possible le "match" K1-K2
         //  en cas de degenerescence renvoie Id
         cSolBasculeRig   SolOfK1K2(int aK1,int aK2,bool & OkSol) const;

         void AssertKValide(int aK) const ;
         void AssertOpened() const ;

         bool                mClosed;
         bool Close(bool Svp= false);
         std::vector<Pt3dr>  mAvant;
         std::vector<Pt3dr>  mApres ;
         std::vector<Pt3dr>  mApresInit ;
         std::vector<Pt3dr>  mSpeedApres ;
         std::vector<std::string>  mNames ;
         bool                mUseV;

         Pt3dr               mP0Avant;
         Pt3dr               mP0Apres;
         double              mLambda;
         double              mDelay;
         double              mCostBestSol;
         cSolBasculeRig      mBestSol;
};

// Resoud par moindre Carre P2 =  mTr + mMatR * aP1 * mL

class cL2EqObsBascult : public cNameSpaceEqF,
                        public cObjFormel2Destroy
{
    public :
        cSolBasculeRig CurSol() const;
        virtual ~cL2EqObsBascult();
        void AddObservation(Pt3dr aP1,Pt3dr aP2,double aPds=1.0,bool WithD2=false); 
    private :
         cL2EqObsBascult(const cL2EqObsBascult&); // N.I.
         friend class cSetEqFormelles;
         void GenCode();

         cL2EqObsBascult
         (
                 cSetEqFormelles &       aSet,
                 const cSolBasculeRig &  aVInit,
                 bool                    Cod2Gen
         );

         cSetEqFormelles *   mSet;
         cRotationFormelle * mRF;

         cEqfBlocIncNonTmp   mBlocL;


         // cIncIntervale       mIntL;
         // double              mLambda;
         // Fonc_Num            mLF;
         
         cP3d_Etat_PhgrF     mN1;
         cP3d_Etat_PhgrF     mN2;
         Pt3d<Fonc_Num>      mResidu;



         std::string           mNameType;
         cIncListInterv        mLInterv;
         cElCompiledFonc *     mFoncEqResidu;
};





#endif //   _EXEMPLE_BASCULEMENT_H_


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
