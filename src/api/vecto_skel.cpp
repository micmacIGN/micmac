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

#define NoOperatorVirgule

#include "StdAfx.h"


// eTypeTapas aTTT = eTT_RRRRRRRRR;

// template <class T> ElSTDNS complex<T> Pt2d2complex(Pt2d<T> p) {return ElSTDNS complex<T>(p.x,p.y);}
ElSTDNS complex<INT> Pt2d2complex(Pt2d<INT> p) {return ElSTDNS complex<INT>(p.x,p.y);}

template <class T> Pt2d<T> complex2Pt2d(ElSTDNS complex<T> p) {return Pt2d<T>(p.real(),p.imag());}

/**********************************************************/
/*                                                        */
/*           EL_CALL_BACK_VECTO                           */
/*                                                        */
/**********************************************************/

EL_CALL_BACK_VECTO::~EL_CALL_BACK_VECTO() {}




/**********************************************************/
/*                                                        */
/*           EL_API_VECTO::ParamSkel                      */
/*                                                        */
/**********************************************************/

EL_API_VECTO::ParamSkel::ParamSkel
              (
                  double Crit_ang,
                  int    Crit_surf,
                  bool   Prolgt_extr,
                  bool   Skel_of_disk
              )  :
                  _crit_ang     (Crit_ang),
                  _crit_surf    (Crit_surf),
                  _prolgt_extr  (Prolgt_extr),
                  _skel_of_disk (Skel_of_disk)
{
}

double EL_API_VECTO::ParamSkel::crit_ang ()    const {return _crit_ang; }
int    EL_API_VECTO::ParamSkel::crit_surf()    const {return _crit_surf;}
bool   EL_API_VECTO::ParamSkel::prolgt_extr()  const {return _prolgt_extr;}
bool   EL_API_VECTO::ParamSkel::skel_of_disk() const {return _skel_of_disk;}



/**********************************************************/
/*                                                        */
/*           EL_API_VECTO::ParamApprox                    */
/*                                                        */
/**********************************************************/

EL_API_VECTO::ParamApprox::ParamApprox
              (
                   double Precision ,
                   int    Anticipation
              ) :
             _precision (Precision),
             _anticipation (Anticipation)
{
}

double EL_API_VECTO::ParamApprox::precision()    const {return _precision;}
int    EL_API_VECTO::ParamApprox::anticipation() const {return _anticipation;}

/**********************************************************/
/*                                                        */
/*           EL_API_VECTO::ParamFonc                      */
/*                                                        */
/**********************************************************/

EL_API_VECTO::ParamFonc::~ParamFonc() {}

EL_API_VECTO::ParamFonc::ParamFonc
              (
                      ComplexI aSz,
                      bool isInferieur ,
                      int  aThreshold 

              )  :
	      mSz (aSz),
	      mInferieur (isInferieur),
	      mThreshold (aThreshold)
{
}

ComplexI EL_API_VECTO::ParamFonc::Sz() const
{
   return mSz;
}


bool EL_API_VECTO::ParamFonc::Inferieur() const    {return mInferieur;}
int  EL_API_VECTO::ParamFonc::Threshold() const    {return mThreshold;}


/**********************************************************/
/*                                                        */
/*           EL_API_VECTO::ParamFile                      */
/*                                                        */
/**********************************************************/

INT EL_API_VECTO::ParamFile::GetThresh
    (
         const std:: string  & aName,
	 INT aTresh
    )
{
    if (aTresh != DefThresh)
        return aTresh;

    GenIm::type_el type = Tiff_Im(aName.c_str()).type_el();
    if (signed_type_num(type))
       return 0;
    else
       return 1 << (nbb_type_num(type)-1);
}

EL_API_VECTO::ParamFile::ParamFile
              (
                      const ElSTDNS string & aName,
                      bool Inferieur ,
                      int  Threshold 

              )  :
	      ParamFonc
	      (
	          Pt2d2complex(Tiff_Im(aName.c_str()).sz()),
		  Inferieur,
		  GetThresh(aName,Threshold)
              ),
              _TiffNameFile (aName)
{
}

Fonc_Num EL_API_VECTO::ParamFile::In(INT aDef)const 
{
   Tiff_Im aTif(_TiffNameFile .c_str());

   return aTif.in(aDef);
}



/**********************************************************/
/*                                                        */
/*           EL_API_VECTO::ParamImage                     */
/*                                                        */
/**********************************************************/


EL_API_VECTO::ParamImage::ParamImage
              (
		  ComplexI aSz,
	          bool     Inferieur,
		  INT      aThreshold,
		  void *   adrEl0,
		  INT      aNbBit,
		  bool     isIntegral,
		  bool     isSigned 

              )  :
	      ParamFonc (aSz,Inferieur,aThreshold),
	      mAdrEl0 (adrEl0),
	      mNBB    (aNbBit),
	      mIntegr (isIntegral),
	      mSigned (isSigned)
{
}

Fonc_Num EL_API_VECTO::ParamImage::In(INT aDef)const 
{
   GenIm::type_el aType = type_im(mIntegr,mNBB,mSigned,true);
   GenIm anIm = alloc_im2d(aType,Sz().real(),Sz().imag(),mAdrEl0);

   return anIm.in(aDef);
}

/**********************************************************/
/*                                                        */
/*           EL_API_VECTO                                 */
/*                                                        */
/**********************************************************/

class EL_API_MAKE_VECTO : public  Br_Vect_Action
{
      public :

          EL_API_MAKE_VECTO
          (
                 EL_CALL_BACK_VECTO   &  call_back,
                 ArgAPP                  arg_app,
                 ComplexI                dec
          )  :
              _call_back   (call_back),
              _arg_app     (arg_app),
              _dec         (complex2Pt2d(dec))
          {
          }

      private :

           void action // => Br_Vect_Action
           (
                   const ElFifo<Pt2di> & pts,
                   const ElFifo<INT>   *,
                   INT
           );

           EL_CALL_BACK_VECTO   &  _call_back;
           ArgAPP                  _arg_app;
           Pt2di                   _dec;

           ElFifo<INT>        _approx;
           ElSTDNS vector<ComplexI>   _pts_interm;
           ElSTDNS vector<int>        _d_interm;

};



void EL_API_MAKE_VECTO::action // => Br_Vect_Action
    (
            const ElFifo<Pt2di> & pts,
            const ElFifo<INT>   * args,
            INT
    )
{
    const  ElFifo<INT> & dist = args[0];
    approx_poly(_approx,pts,_arg_app);

    for (INT kAp=1 ; kAp< _approx.nb() ; kAp++)
    {
         INT k1 = _approx[kAp-1];
         INT k2 = _approx[kAp];

         _pts_interm.clear();
         _d_interm.clear();

         for (INT k=k1 ; k<= k2 ; k++)
         {
              _pts_interm.push_back(Pt2d2complex(pts[k]+_dec));
              _d_interm.push_back(dist[k]);
         }

         _call_back.ElVectoAction
         (
              Pt2d2complex(pts[k1]+_dec),
              Pt2d2complex(pts[k2]+_dec),
              _pts_interm,
              _d_interm
         );
    }

}





EL_API_VECTO::EL_API_VECTO
(
       ParamSkel   PSkel,
       ParamApprox PApprox,
       INT          Max_Dist,
       bool         LocCoord 
)  :
   _pskel       (PSkel),
   _papprox     (PApprox),
   _max_dist    (Max_Dist),
   _LocCoord    (LocCoord)
{
}

const ComplexI  EL_API_VECTO::DefP0(0,0);
const ComplexI  EL_API_VECTO::DefSZ(-0x7fffffff,-0x7fffffff);


void EL_API_VECTO::vecto
     (
                    const ParamFonc &        PFile,
                    EL_CALL_BACK_VECTO  &    Call_back,
                    ComplexI                 p0,
                    ComplexI                 sz 
     )
{

      if (sz == DefSZ)
         sz = PFile.Sz();

      int thr = PFile.Threshold();

      Fonc_Num f =  PFile.Inferieur()         ?
                    (PFile.In(thr)   < thr)    :
                    (PFile.In(thr-1) >= thr)   ;


     
     ELISE_COPY
     (
          rectangle(complex2Pt2d(p0),complex2Pt2d(sz)+complex2Pt2d(p0)),
          sk_vect
          (
                  skeleton_and_dist
                  (
                         f,
                         _max_dist,
                         L_ArgSkeleton()
                      +  ArgSkeleton(SurfSkel(_pskel.crit_surf()))
                      +  ArgSkeleton(AngSkel(_pskel.crit_ang()))
                      +  ArgSkeleton(ProlgtSkel(_pskel.prolgt_extr()))
                      +  ArgSkeleton(SkelOfDisk(_pskel.skel_of_disk()))
                  ),
                 new  EL_API_MAKE_VECTO 
                      (
                            Call_back,
                            ArgAPP
                            (   
                                   _papprox.precision(),
                                   _papprox.anticipation(),
                                   ArgAPP::D2_droite,
                                   ArgAPP::MeanSquare
                            ),
                           _LocCoord ? -p0 : ComplexI(0,0)
                      )
          ),
          Output::onul()
     );
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
