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


double GETVALPT(Fonc_Num aF,Pt2di aP)
{
   double aSom;
   ELISE_COPY(rectangle(aP,aP+Pt2di(1,1)),Rconv(aF),sigma(aSom));
   return aSom;
}

/*********************************************************************/
/*                                                                   */
/*         Arg_Fonc_Num_Comp                                         */
/*                                                                   */
/*********************************************************************/

Arg_Fonc_Num_Comp::Arg_Fonc_Num_Comp(Flux_Pts_Computed * flx) :
      _flux (flx)
{
}


/***************************************************************/
/*                                                             */
/*          Fonc_Num_Computed                                  */
/*                                                             */
/***************************************************************/

Fonc_Num_Computed::~Fonc_Num_Computed() {}


Fonc_Num_Computed::Fonc_Num_Computed
   (  const Arg_Fonc_Num_Comp & arg,
      INT                        dim_out,
      Pack_Of_Pts::type_pack type_out
   ) :
     _dim_out    (dim_out),
     _type_out   (type_out),
     _flux_of_comp (arg.flux())
{
}

bool  Fonc_Num_Computed::icste(INT *) {return false;}

bool Fonc_Num_Computed::integral () const
{
     switch (_type_out)
     {
            case  Pack_Of_Pts::integer : return true;
            case  Pack_Of_Pts::real :    return false;
            default :
                     elise_internal_error("ImOutNotComp::compute",__FILE__,__LINE__);
     }

     return 0;

}

/***************************************************************/
/*                                                             */
/*          Fonc_Num                                           */
/*                                                             */
/***************************************************************/

#define VERIF_INTEGRAL_FONC 1
#define VERIF_DIM_FONC_OUT  1

Fonc_Num::Fonc_Num(void) :
     PRC0()
{
}

Fonc_Num::Fonc_Num(Fonc_Num_Not_Comp * fonc) :
     PRC0(fonc)
{
}

Fonc_Num_Computed * Fonc_Num::compute(const Arg_Fonc_Num_Comp & arg)
{

  Fonc_Num_Computed * fnc = SAFE_DYNC(Fonc_Num_Not_Comp *,_ptr)->compute(arg);

#if (VERIF_INTEGRAL_FONC)
El_Internal.ElAssert
(
     (fnc->integral() ==integral_fonc(arg.flux()->integral_flux())),
     EEM0 << "Incohrence in integral fonc"
);
#endif // VERIF_INTEGRAL_FONC

#if (VERIF_DIM_FONC_OUT)
El_Internal.ElAssert
(
     (dimf_out() == fnc->idim_out()),
     EEM0 << "Incohrence in verif dim out " 
          << "[fonc:" << dimf_out()  << "] " 
          << " [comp : " << fnc->idim_out() << " ]\n"
);
#endif // VERIF_DIM_FONC_OUT

  return fnc;
}

bool  Fonc_Num::integral_fonc(bool iflux) const
{
   return SAFE_DYNC(Fonc_Num_Not_Comp *,_ptr)->integral_fonc(iflux);
}

INT  Fonc_Num::dimf_out() const
{
   return SAFE_DYNC(Fonc_Num_Not_Comp *,_ptr)->dimf_out();
}

void  Fonc_Num::VarDerNN(ElGrowingSetInd & aSet) const
{
      SAFE_DYNC(Fonc_Num_Not_Comp *,_ptr)->VarDerNN(aSet);
}


const Fonc_Num Fonc_Num::No_Fonc;

bool  Fonc_Num::really_a_fonc() const
{
      return _ptr != 0;
}


INT Fonc_Num::CmpFormel(const Fonc_Num & aF2) const
{
   Fonc_Num_Not_Comp * Ptr1 = SAFE_DYNC(Fonc_Num_Not_Comp *,_ptr);
   Fonc_Num_Not_Comp * Ptr2 = SAFE_DYNC(Fonc_Num_Not_Comp *,aF2._ptr);


   INT cmp = CmpTertiare(Ptr1->KindOfExpr(),Ptr2->KindOfExpr());
   if (cmp) 
      return cmp;

   return  Ptr1->CmpFormelIfSameKind(Ptr2);
}

bool Fonc_Num::IsVarSpec(void) const
{
  return  KindOfExpr() ==  Fonc_Num::eIsVarSpec;
}

bool  Fonc_Num::IsAtom() const
{
   tKindOfExpr aK = KindOfExpr();

   return    (aK==eIsICste )
          || (aK==eIsRCste )
          || (aK==eIsFCoord)
          || (aK==eIsVarSpec);
}
bool  Fonc_Num::IsOpbin() const
{
  return  KindOfExpr() ==  Fonc_Num::eIsOpBin;
}
bool  Fonc_Num::IsOpUn() const
{
  return  KindOfExpr() ==  Fonc_Num::eIsOpUn;
}

Fonc_Num::tKindOfExpr  Fonc_Num::KindOfExpr() const
{
   return  SAFE_DYNC(Fonc_Num_Not_Comp *,_ptr)->KindOfExpr();
}


REAL Fonc_Num::ValFonc(const class PtsKD & pts) const
{
     return SAFE_DYNC(Fonc_Num_Not_Comp *,_ptr)->ValFonc(pts);
}

Fonc_Num Fonc_Num::deriv(INT kth) const
{
     return SAFE_DYNC(Fonc_Num_Not_Comp *,_ptr)->deriv(kth);
}

void  Fonc_Num::compile(cElCompileFN & anEnv)
{
     SAFE_DYNC(Fonc_Num_Not_Comp *,_ptr)->compile(anEnv);

     anEnv.AddToDict(*this);
}



REAL Fonc_Num::ValDeriv(const PtsKD &pts,INT kth) const
{
     return SAFE_DYNC(Fonc_Num_Not_Comp *,_ptr)->ValDeriv(pts,kth);
}

INT  Fonc_Num::DegrePoly() const
{
     return SAFE_DYNC(Fonc_Num_Not_Comp *,_ptr)->DegrePoly();
}

INT Fonc_Num::NumCoord() const
{
    return  SAFE_DYNC(Fonc_Num_Not_Comp *,_ptr)->NumCoord();
}


Fonc_Num  Fonc_Num::Simplify() 
{
   Fonc_Num_Not_Comp * pFNNC = SAFE_DYNC(Fonc_Num_Not_Comp *,_ptr);

    return pFNNC->Simplify();

}


void Fonc_Num::show(std::ostream & os) const
{
     Fonc_Num_Not_Comp * pFNNC = SAFE_DYNC(Fonc_Num_Not_Comp *,_ptr);
     pFNNC->show(os);
}

bool  Fonc_Num::is0() const
{
     return SAFE_DYNC(Fonc_Num_Not_Comp *,_ptr)->is0();
}


void  Fonc_Num::inspect() const
{
     SAFE_DYNC(Fonc_Num_Not_Comp *,_ptr)->inspect();
}


bool  Fonc_Num::is1() const
{
     return SAFE_DYNC(Fonc_Num_Not_Comp *,_ptr)->is1();
}

bool  Fonc_Num::IsCsteRealDim1(REAL &aVal) const
{
     return SAFE_DYNC(Fonc_Num_Not_Comp *,_ptr)->IsCsteRealDim1(aVal);
}

std::string Fonc_Num::NameCpp()
{
    return SAFE_DYNC(Fonc_Num_Not_Comp *,_ptr)->NameCpp();
}
bool Fonc_Num::HasNameCpp()
{
    return SAFE_DYNC(Fonc_Num_Not_Comp *,_ptr)->HasNameCpp();
}
void  Fonc_Num::SetNameCpp(const std::string & aName)
{
	SAFE_DYNC(Fonc_Num_Not_Comp *,_ptr)->SetNameCpp(aName);
}


bool  Fonc_Num::IsCsteRealDim1(INT &aVal) const
{
     REAL aRV;
     bool res = IsCsteRealDim1(aRV);
     aVal = round_ni(aRV);
     return res;
}

REAL Fonc_Num::ValFoncGen(Pt2di aP) const
{
    REAL aSom;
    ELISE_COPY
    (
       rectangle(aP,aP+Pt2di(1,1)),
       Rconv(*this),
       sigma(aSom)
    );
    return aSom;
}

REAL Fonc_Num::ValFoncGenR(Pt2dr aPR) const
{
    Pt2di  aPI(round_down(aPR.x),round_down(aPR.y));
    static  Im2D<REAL,REAL> anIm(2,2);
    static  TIm2D<REAL,REAL>  aTim(anIm);

    ELISE_COPY
    (
       anIm.all_pts(),
       trans(Rconv(*this),aPI),
       anIm.out()
    );
    Pt2dr aPFrac(aPR.x-aPI.x,aPR.y-aPI.y);
    return aTim.getr(aPFrac);
}



/***************************************************************/



/***************************************************************/
/*                                                             */
/*          Fonc_Num_Not_Comp                                  */
/*                                                             */
/***************************************************************/
Fonc_Num Fonc_Num_Not_Comp::deriv(INT) const
{
      ELISE_ASSERT(false,"No deriv for this func");
      return 0;
}

void  Fonc_Num_Not_Comp::compile(cElCompileFN & anEnv)
{
      ELISE_ASSERT(false,"No compile for this func");
}

Fonc_Num::tKindOfExpr  Fonc_Num_Not_Comp::KindOfExpr()
{
   ELISE_ASSERT(false,"No KindOfExpr for this func");
   return Fonc_Num::eIsUnknown;
}

INT Fonc_Num_Not_Comp::CmpFormelIfSameKind(Fonc_Num_Not_Comp *)
{
   ELISE_ASSERT(false,"No CmpFormelIfSameKind for this func");
   return 0;
}

Fonc_Num Fonc_Num_Not_Comp::Simplify() 
{
   
   return Fonc_Num(const_cast<Fonc_Num_Not_Comp *>(this));
}

void  Fonc_Num_Not_Comp::show(std::ostream &) const
{
      ELISE_ASSERT(false,"No show for this func");
}

REAL  Fonc_Num_Not_Comp::ValFonc(const  PtsKD &) const
{
      ELISE_ASSERT(false,"No Value this func");
      return 0;
}                

REAL  Fonc_Num_Not_Comp::ValDeriv(const  PtsKD &,INT) const
{
      ELISE_ASSERT(false,"No deriv for this func");
      return 0;
}

Fonc_Num_Not_Comp::Fonc_Num_Not_Comp() :
    mNameCPP (0)
{
}
Fonc_Num_Not_Comp::~Fonc_Num_Not_Comp() 
{
	delete mNameCPP;
}

std::string Fonc_Num_Not_Comp::NameCpp()
{
    return mNameCPP ? *mNameCPP : std::string("");
}

bool Fonc_Num_Not_Comp::HasNameCpp()
{
    return mNameCPP != 0;
}

void  Fonc_Num_Not_Comp::SetNameCpp(const std::string & aName)
{
      delete mNameCPP;
      mNameCPP = new std::string(aName);
}

INT  Fonc_Num_Not_Comp::DegrePoly() const
{
    return -1;
}

INT Fonc_Num_Not_Comp::NumCoord() const
{
      ELISE_ASSERT(false,"NumCoord is limited to Fonc Coordinates");
      return 0;
}

bool Fonc_Num_Not_Comp::is0() const
{
     return false;
}

void Fonc_Num_Not_Comp::inspect() const
{
}

bool Fonc_Num_Not_Comp::is1() const
{
     return false;
}

bool Fonc_Num_Not_Comp::IsCsteRealDim1(REAL &) const
{
     return false;
}






/***************************************************************/
/*                                                             */
/*          Op_Un_Not_Comp                                     */
/*                                                             */
/***************************************************************/


Op_Un_Not_Comp::Op_Un_Not_Comp
(
     Fonc_Num          f,
     const char *      name,
     TyVal             Value,
     TyDeriv           Deriv,
     TyValDeriv        ValDeriv
)  :
    _f (f),
    _name (name),
    _OpUnVal (Value),
    _OpUnDeriv (Deriv),
    mOpUnValDeriv (ValDeriv)
{
}




Fonc_Num_Computed * Op_Un_Not_Comp::compute(const Arg_Fonc_Num_Comp & arg)
{
    Fonc_Num_Computed *fc;

    fc = _f.compute(arg);

    return op_un_comp(arg,fc);
}

INT Op_Un_Not_Comp::dimf_out() const
{
    return _f.dimf_out();
}

void Op_Un_Not_Comp::VarDerNN(ElGrowingSetInd & aSet) const
{
    _f.VarDerNN(aSet);
}

/***************************************************************/
/*                                                             */
/*          Op_Bin_Not_Comp                                    */
/*                                                             */
/***************************************************************/

INT Op_Bin_Not_Comp::dimf_out() const
{
    return ElMax
           (
               _f0.dimf_out(),
               _f1.dimf_out()
           );
}

void Op_Bin_Not_Comp::VarDerNN(ElGrowingSetInd & aSet) const
{
    _f0.VarDerNN(aSet);
    _f1.VarDerNN(aSet);
}


Op_Bin_Not_Comp::Op_Bin_Not_Comp
(
     Fonc_Num          f0,
     Fonc_Num          f1,
     bool              isInfixe,
     const char *      name,
     TyVal             Value,
     TyDeriv           Deriv,
     TyValDeriv        ValDeriv
)  :
    _f0             (f0),
    _f1             (f1),
    mIsInfixe       (isInfixe),
    _name           (name),
    _OpBinVal       (Value),
    _OpBinDeriv     (Deriv),
    mOpBinValDeriv  (ValDeriv)
{
}

Fonc_Num_Computed * Op_Bin_Not_Comp::compute(const Arg_Fonc_Num_Comp & arg)
{
    Fonc_Num_Computed *fc0,*fc1;

    fc0 = _f0.compute(arg);
    fc1 = _f1.compute(arg);

    Tjs_El_User.ElAssert
    (
            (fc0->idim_out() == fc1->idim_out())
         || (fc0->idim_out() == 1)
         || (fc1->idim_out() == 1)  ,
         EEM0 <<  " incompatible dimensions in binary operator \""
              << _name  << "\"\n"
              <<  "|          "
              <<  " (dim1 = " << fc0->idim_out()
              <<  " , dim2 = " << fc1->idim_out() << ")"
    );

    return op_bin_comp(arg,fc0,fc1);
}


/***************************************************************/
/*                                                             */
/*          Convertion of a set of function to a common type   */
/*                                                             */
/***************************************************************/

/*
    If one of the tf is REAL, all will be converted to REAL.
*/


Pack_Of_Pts::type_pack  convert_fonc_num_to_com_type
     (
          const Arg_Fonc_Num_Comp & arg,
          Fonc_Num_Computed * * tf,
          Flux_Pts_Computed * flx,
          INT nb
     )
{
     Pack_Of_Pts::type_pack type_res =  Pack_Of_Pts::integer;


     for(int i = 0; i<nb ; i++)
        if (tf[i]->type_out() ==  Pack_Of_Pts::real)
           type_res =  Pack_Of_Pts::real;

	 // change the name from i to j because of fucking Visual C--
     for(int j = 0; j<nb ; j++)
        tf[j] = convert_fonc_num(arg,tf[j],flx,type_res);

     return type_res;
}

Fonc_Num SuperiorStrict(Fonc_Num f1,Fonc_Num f2)
{
     return f1 > f2;
}

Fonc_Num NotEqual(Fonc_Num f1,Fonc_Num f2)
{
     return f1 != f2;
}



/*
Fonc_Num Virgule(Fonc_Num f1,Fonc_Num f2)
{
     return (f1 , f2);
}
*/

Fonc_Num Virgule(Fonc_Num f1,Fonc_Num f2,Fonc_Num f3)
{
     return Virgule(f1 , Virgule(f2 , f3));
}

Fonc_Num Virgule(Fonc_Num f1,Fonc_Num f2,Fonc_Num f3,Fonc_Num f4)
{
     return Virgule(f1,Virgule(f2,f3,f4));
}

Fonc_Num Virgule(Fonc_Num f1,Fonc_Num f2,Fonc_Num f3,Fonc_Num f4,Fonc_Num f5)
{
     return Virgule(f1,Virgule(f2,f3,f4,f5));
}


Fonc_Num Virgule(Fonc_Num f1,Fonc_Num f2,Fonc_Num f3,Fonc_Num f4,Fonc_Num f5,Fonc_Num f6)
{
     return Virgule(f1,Virgule(f2,f3,f4,f5,f6));
}

/***************************************************************/
/*                                                             */
/*          PtsKD                                              */
/*                                                             */
/***************************************************************/

void PtsKD::init(int Dim)
{
   _dim = Dim; 
   _x = NEW_VECTEUR(0,Dim,REAL);
    MEM_RAZ(_x,_dim);
}

PtsKD::PtsKD(INT Dim) 
{
    init(Dim);
}

PtsKD::PtsKD(REAL * v,INT Dim) 
{
    init(Dim);
    convert(_x,v,Dim);
}

PtsKD::~PtsKD()
{
  DELETE_VECTOR(_x,0);
}

PtsKD::PtsKD(const PtsKD & p)
{
    init(p._dim);
    convert(_x,p._x,_dim);
}

void PtsKD::operator = (const PtsKD & p)
{
    DELETE_VECTOR(_x,0);
    init(p._dim);
    convert(_x,p._x,_dim);
}

/*
void PtsKD::Set(Fonc_Num aF,REAL aV)
{
   (*this)(aF.NumCoord()) = aV;
}

void PtsKD::Set(const Pt3d<Fonc_Num> & aPF,const Pt3d<REAL>  & aPR)
{
     Set(aPF.x,aPR.x);
     Set(aPF.y,aPR.y);
     Set(aPF.z,aPR.z);
}

void PtsKD::Set(const TplElRotation3D<Fonc_Num> & aRF,const TplElRotation3D<REAL>  & aRR)
{
    Set(aRF.tr(),aRR.tr());
    Set(aRF.teta01(),aRR.teta01());
    Set(aRF.teta02(),aRR.teta02());
    Set(aRF.teta12(),aRR.teta12());
}

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
