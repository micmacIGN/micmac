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



#ifndef  _ELISE_API_VECTO_H
#define  _ELISE_API_VECTO_H

#include <vector>
#include <list>
#include <string>
#include <complex>


class  Fonc_Num;
template <class  Type> class Pt2d;
template <class  Type> class Box2d;
template <class  Type> class ElList;
typedef std::complex<int> ComplexI;
typedef std::complex<double> ComplexR;
class ElHough;
class Seg2d;


// 2 Fonctions utiles en support a l'implementation

/*
Pt2d<int> Std2Elise(const ComplexI &);
ComplexI  Elise2Std(const Pt2d<int> &);

Pt2d<double> Std2Elise(const ComplexR &);
ComplexR  Elise2Std(const Pt2d<double> &);
*/



inline Pt2di Std2Elise(const ComplexI & aComp) { return Pt2di(aComp.real(),aComp.imag()); }
inline Pt2dr Std2Elise(const ComplexR & aComp) { return Pt2dr(aComp.real(),aComp.imag()); }
inline ComplexI  Elise2Std(const Pt2d<int> & aPt) { return ComplexI(aPt.x,aPt.y); }
inline ComplexR  Elise2Std(const Pt2d<double> & aPt) { return ComplexR(aPt.x,aPt.y); }




template <class Type>  
ElList<Type> TplStd2Elise(const std::list<Type> & aL)
{
   ElList<Type> aRes;

   for 
   (
        typename std::list<Type>::const_iterator   itP=aL.begin();
        itP!=aL.end();
        itP++
   )
       aRes = aRes + *itP;

   return aRes;
}



/************************************************************************/
/************************************************************************/
/****                                                                ****/
/****                                                                ****/
/****            VECTORISATION vs SQUELETISATION                     ****/
/****                                                                ****/
/****                                                                ****/
/************************************************************************/
/************************************************************************/



class EL_CALL_BACK_VECTO
{
      public :
          virtual void  ElVectoAction
                        (
                              ComplexI p0,
                              ComplexI p1,
                              const std:: vector<ComplexI> & pts_interm,
                              const std:: vector<int> & d_interm
                        ) = 0;

          virtual ~EL_CALL_BACK_VECTO();
      private :
};


class EL_API_VECTO
{
      public :

           class ParamSkel
           {
                 public :

                    ParamSkel
                    (
                          double Crit_ang     = 3.14,
                          int    Crit_surf    = 8,
                          bool   Prolgt_extr  = true,
                          bool   Skel_of_disk = false
                    );

                    double crit_ang      ()    const ;
                    int    crit_surf     ()    const ;
                    bool   prolgt_extr   ()    const ;
                    bool   skel_of_disk  ()    const ;

                 private :

                          double  _crit_ang;
                          int     _crit_surf;
                          bool    _prolgt_extr;
                          bool    _skel_of_disk;
           };

          class ParamApprox
          {
                public :
                
                   // precision : + c'est fort, moins y'a de points
                   // anticipation : + c'est fort + c'est long (cpu) 
                   //      et meilleure est la qualite du resultat

                   ParamApprox
                   (
                         double Precision = 10.0, 
                         int    Anticipation = 10  
                   );
                    double precision()     const ;
                    int    anticipation () const ;

                private :
                     double _precision;
                     int    _anticipation;
          }; 

           class ParamFonc
           {
                public :
                    ComplexI Sz() const;
                    bool  Inferieur() const;
                    int   Threshold() const;
                    virtual  Fonc_Num In(int aDef)const  = 0;
                    virtual  ~ParamFonc();
                protected :
                    ParamFonc
		    (
                           ComplexI aSz,
                           bool     Inferieur ,
                           int       Threshold 
		    );
                private :
                    ComplexI       mSz;
                    bool          mInferieur;
                    int           mThreshold;
           };

           class ParamFile : public ParamFonc
           {
                public :


                   // si inferieur = true, considere l'image binaire des points
                   // tels que  ValeurPixel < threshold
                   // sinon ceux tels que ValeurPixel >= threshold

                   // Pour la valeur par defaut de threshold , le seuil
                   // correspond a la moitie de la valeur max theorique du fichier
                   // tiff (typiquement 1 pour une image binaire, 128 pour
                   // une image sur un octet) pour un fichier non signe
                   // et a 0 pour un fichier signe

                   enum {
                        DefThresh = -0x7fffffff
                   };

                    ParamFile
                    (
                           const std:: string & TIFFNameFile,
                           bool Inferieur = true,
                           int  Threshold = DefThresh
                    );


                private :

                    virtual  Fonc_Num In(int aDef)const ;
                    const std:: string  _TiffNameFile;
		    static int GetThresh(const std:: string  & aName,int aTresh);
           };


           class ParamImage : public ParamFonc
	   {
                public :
		   ParamImage
		   (
		        ComplexI aSz,
		        bool     Inferieur,
		        int      aThreshold,
			void *   adrEl0,
			int      aNbBit,
			bool     isIntegral = true,
			bool     isSigned = false
		   );
                private :
                    virtual  Fonc_Num In(int aDef)const ;
		    void * mAdrEl0;
		    int mNBB;
		    bool mIntegr;
		    bool mSigned;
	   };


          EL_API_VECTO 
          (
                   ParamSkel,
                   ParamApprox,
                   int   max_dist = 256,
                   bool  LocCoord = true 
          );
             // LocCoord :  Si true le point de vecto de
             // coordonnees   0,0 correspond  l'origine
             // des zones vecto, sinon a l'origine fichier


          static const ComplexI  DefP0;
          static const ComplexI  DefSZ;



          void vecto
               (
                    const ParamFonc &        PFile,
                    EL_CALL_BACK_VECTO  &    Call_back,
                    ComplexI                 p0 = DefP0,
                    ComplexI                 sz = DefSZ
               );

          //  DefP0 => origine du fichier (0,0)
          //  DefSZ => taille  du fichier

         
      private :

          ParamSkel      _pskel;
          ParamApprox   _papprox;
          int           _max_dist;
          bool          _LocCoord;

};


/************************************************************************/
/************************************************************************/
/****                                                                ****/
/****                                                                ****/
/****            VECTORISATION vs HOUGH                              ****/
/****                                                                ****/
/****                                                                ****/
/************************************************************************/
/************************************************************************/


class HoughMapedParam
{
     public :

         // Les trois valeur a donner pour initialiser la structure de
         // parametrisation sont approximatives.


          HoughMapedParam
          (
               const std::string  & NameHough,
               double aLengthMinSeg,   // Valeur Min des segmen recherches [1]
               double aLengthMaxSeg,   // Valeur Max des segmen recherches [1]
               double aVminRadiom,    // typiquement 255 pour images  SAR, pre-filtree
               bool aModeSar        // true : genre SAR, false genre gradient
          );


           Fonc_Num FiltreSAR(Fonc_Num fIm,Fonc_Num fInside);
           Box2d<int> BoxRab()   ;
           Pt2d<int>  SzHoughUti() ;
           Pt2d<int>  P0HoughUti() ;
           Pt2d<int>  P1HoughUti() ;
           ElHough &  Hough();


      // Seuils "Fondamentaux"

         std::string         mNameHough;
         double              mLengthMinSeg;  // LenthMinSeg
         double              mLengthMaxSeg;  // LenthMaxSeg
         double              mVminRadiom;
         int                 mModeH;



      // Seuil pour le prefiltrage des images


        // Parametre, specifiques au mode gradient qui
        // permettent de faire aparaitre des 0 avant hough
        // (Utile pour accelerer et, eventuellement, debruiter)
        // Les valeurs par defaut peuvent convenir


         double  mFactCaniche;    // Facteur de Canny-Deriche , Default = 1.0
         bool    mGradUseFiltreMaxLocDir; // utilise-t-on le filtre "MaxLocDir" avant Hough , Def = true
         bool    mGradUseSeuilPreFiltr; // utilise-t-on la valeur mGradSeuilPreFiltr comme seuil avant Hough; Def = true
         double    mGradSeuilPreFiltr;    // Def = mVminRadiom


         // Parametre specifique au Mode Sar, 
         // ** =  valeur par defaut devant  certainement etre reconsideree  par l'appli
         // *  =  valeur par defaut devant probalement etre reconsideree  par l'appli

         int   mSAR_NbVoisMoyCFAR;   // ** defaut = 10
         int   mSAR_NbStepMoyCFAR;   // ** defaut = 4
         int   mSAR_NbVoisMedian;    // * defaut = 1
         double  mSAR_FactExpFilter;   // *  defaut = 0.4
         int   mSAR_BinariseSeuilBas;  // defaut = 180 
               // Ces seuils sont a peu pres portable car ils operent sur des images normalisees
         int   mSAR_BinariseSeuilHaut;  // defaut =  230
         int   mSAR_SzConnectedCompMin; // defaut = 2 *  aLengthMinSeg


     // Seuils Utilise pour le filtrage classique des maxima de la transformee
     // de Hough


         // Taille des voisinage en Rho et Teta qui definissent les maxima locaux
         double    mVoisRhoMaxLocInit;    //  Def = 2.0
         double    mVoisTetaMaxLocInit;   //  en radian, Def = 0.1  

        // Seuils utilise pour filtrer les maxima locaux
         double    mSeuilAbsHough;   // aLengthMinSeg * aVminRadiom * 0.7


    //  Seuils Utilises pour le filtrage supplementaire par "Bande-Connectee" 

         bool  mUseBCVS;  // Def = true si mode grad  et false sinon (on ne fait pas de filtrage)

         double  mFiltrBCVSFactInferior;  // Def = 0.5, on redescend juqu'a la moitie
         double  mVoisRhoMaxLocBCVS;      // Def =  mVoisRhoMaxLocInit * 2.5
         double  mVoisTetaMaxLocBCVS;     // Def = mVoisTetaMaxLocInit * 2.5



     // Seuil utilises pour la transformation des droites en segments 

     //
         double mDistMinPrgDyn; //  LenthMinSeg


         double mFiltrSzHoleToFill;  // LenthMinSeg
         double mFiltrSeuilRadiom; // Def = mVminRadiom

          // Specifique aux images de gradient

         double    mHoughIncTeta;   // 1/2 largeur des intervalles angulaires  sur lesquels on accumule; Def=0.2
         double    mFiltrIncTeta;   // Def = mHoughIncTeta * 1.5
         double    mFiltrEcMaxLoc;  // Cout En Pixel d'un ecart / au maxima local, Def = 1.0


      // Seuil pour une post (pre) optimization de la geometrie des segment 
      // (pre = avant decoupe par prg-dyn, post = tout a la fin)
         bool     mDoPreOptimSeg; // def = false, car a dangereux 
                                  // (avant decoupe influence de l'environnement)
         double     mPrecisionPreOptimSeg; //   def = 0.1
         bool     mDoPostOptimSeg; //         def = true
         double     mPrecisionPostOptimSeg; //  def = 0.01


     //  Seuils utilises pour la suppression a posteriori des redondances

         double mLongFiltrRedond;   // LenthMinSeg
         double mLargFiltrRedond;   // LenthMinSeg


     // Divers

         double mRatioRecouvrt; // Ratio de recouvrement entre les accumulations
                              // Def = 0.5; (avec cette valeur chaque point passe par 4 accumulateurs)


     // Abscons et non commentes; La valeur par defaut doit le faire tres bien

         double mDeltaMinExtension;      // Def = 1.0 
         double mFiltrBCVSFactTolGeom;   // Def = 2 
         double mStepInterpPrgDyn;       // Def = 1.0
         double mWidthPrgDyn;            // Def = 3.0 (-> MPD ? voir si 1.0 le fait pas tres bien)

    //  VARIABLE STATIC 

       static double DefVoisRhoMaxLocInit () {return 2.0;}
       static double DefVoisTetaMaxLocInit() {return 0.10;}
       static double RatioDefSeuilAbsHough() {return 0.7;}


       static double DefBCVSFactInf()         {return 0.5;}
       static double DefRatioBCVS()           {return 2.5;}
       static double DefFactCaniche()         {return 1.0;}
       static double DefHoughIncTeta()        {return 0.2;}
       static double DefRatioFiltrIncTeta()   {return 1.5;}
       static double DefEcMaxLoc()            {return 1.0;}
       static double DefRatioRecouvrt()       {return 0.5;}


       static bool DefGradUseFiltreMaxLocDir() { return true;}
       static bool DefGradUseSeuilPreFiltr()   { return true;}

       
         static ElHough *        theStdH;
         static ElHough & HoughOfString(const std::string &);


          Fonc_Num Filtr_CFAR(Fonc_Num fIm,Fonc_Num fPds);

          Fonc_Num FiltrMedian(Fonc_Num fPds);
          Fonc_Num FiltrCanExp(Fonc_Num fIm,Fonc_Num fPds);
          Fonc_Num FiltrBinarise(Fonc_Num fIm);
          Fonc_Num FiltrConc(Fonc_Num fIm);

     private :

         HoughMapedParam(const HoughMapedParam &);

};





/*
    [1] La signification de ce parametre est "on ne donne
        aucune garantie sur le resultat pour les segement qui s'avereraient
        etre de longueur superieure a LenthMaxSeg".

        Il est, entre autre, possible que des segment de longueur superieure a LenthMaxSeg
        soient trouves. 


*/


class HoughMapedInteractor
{
     public :
         std::vector<ComplexR> & Extr0() {return mExtr0;};
         std::vector<ComplexR> & Extr1() {return mExtr1;};
         virtual void OnNewSeg(const ComplexR &,const ComplexR &);
         virtual void OnNewCase(const ComplexI &,const ComplexI &);
         virtual ~HoughMapedInteractor(); 

     private :
          std::vector<ComplexR>  mExtr0;
          std::vector<ComplexR>  mExtr1;
};


void HoughMapFromFile
     (
          const std::string &,    // NOM DU FICHIER
          const ComplexI & p0, const ComplexI & p1,     // rectangle image
          HoughMapedParam &,      // Parametre de detection de segment
          HoughMapedInteractor &  // Interacteur 
     );

void HoughMapFromImage
     (
          unsigned char ** im ,    int Tx, int Ty,  // PARAMETRE IMAGE
          const ComplexI & p0, const ComplexI & p1,     // rectangle image
          HoughMapedParam &,      // Parametre de detection de segment
          HoughMapedInteractor &  // Interacteur 
     );


#endif //  _ELISE_API_VECTO_H


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
