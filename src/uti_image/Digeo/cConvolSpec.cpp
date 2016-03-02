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

#include "cConvolSpec.h"
#include "debug.h"

#include <limits>
#include <sstream>
#include <fstream>
#include <cctype>
#include <cstring>
#include <cmath>
#include <cstdlib>

// __DEL
#include <iostream>

using namespace std;

#define EPSILON_SYMMETRY 1e-6
#define EPSILON_MATCH 1e-4

// Permt de shifter les entiers (+ rapide que la div) sans rien faire pour
// les flottants
inline REAL ShiftDr(const REAL & aD,const int &) { return aD; }
inline REAL ShiftG(const REAL & aD,const int &) { return aD; }
inline REAL InitFromDiv(REAL,REAL *) { return 0; }

inline INT ShiftDr(const INT & aD,const int & aShift) { return aD >> aShift; }
inline INT ShiftG(const INT & aD,const int & aShift) { return aD << aShift; }
inline INT InitFromDiv(INT aDiv,INT *) { return aDiv/2; }

template <class Type> inline Type Abs( Type v1 ) { return ( (v1>0) ? v1 : -v1 ); }

//----------------------------------------------------------------------
// cConvolSpec methods
//----------------------------------------------------------------------

/*
template <class Type>
class cSomFiltreSep : public Simple_OPBuf1<Type,Type>
{
    public :
        cSomFiltreSep( cConvolSpec<Type> * );
 
         void ShowLine(Type * aLine, Video_Win * aW)
         {
             int aSzX = this->x1() - this->x0();
             Im2D<Type,Type> anIm(aSzX,1);
             for (int anX = this->x0(); anX<this->x1() ; anX++)
             {
                  anIm.data()[0][anX-this->x0()] = aLine[anX];
             }
             int anY = this->ycur();
             ELISE_COPY
             (
                 rectangle(Pt2di(this->x0(),anY),Pt2di(this->x1(),anY+1)),
                 trans(anIm.in(),-Pt2di(this->x0(),anY)),
                 aW->ogray()
             );
         }

    private :
       void  calc_buf (Type  ** output,Type  *** input);

        
        void ConvolLine(int y);

        cConvolSpec<Type> * mConvol;
        ~cSomFiltreSep();
        int                 mDeb;
        int                 mFin;
        Type  **            mLineFiltered;
        Type  **            mInPut;

        Simple_OPBuf1<Type,Type> * dup_comp();
};

template <class Type> cSomFiltreSep<Type>::cSomFiltreSep(cConvolSpec<Type> * aCS ):
    mConvol  (aCS),
    mDeb     (aCS->Deb()),
    mFin     (aCS->Fin()),
    mLineFiltered (0),
    mInPut          (0)
{
    
}

template <class Type> cSomFiltreSep<Type>::~cSomFiltreSep()
{
   if (mLineFiltered)
   {
       DELETE_MATRICE(mLineFiltered,Pt2di(this->x0Buf(),this->y0Buf()),Pt2di(this->x1Buf(),this->y1Buf()));
   }
}

template <class Type> Simple_OPBuf1<Type,Type> * cSomFiltreSep<Type>::dup_comp()
{
   cSomFiltreSep<Type> * aRes = new cSomFiltreSep<Type>(mConvol);

   Pt2di aP0(this->x0Buf(),this->y0Buf());
   Pt2di aP1(this->x1Buf(),this->y1Buf());

   aRes->mLineFiltered =  NEW_MATRICE(aP0,aP1,Type);

   return aRes;
}

template <class Type> void cSomFiltreSep<Type>::ConvolLine(int y)
{

   mConvol->Convol(mLineFiltered[y],mInPut[y],this->x0(),this->x1());
}

template <class Type> void cSomFiltreSep<Type>::calc_buf (Type  ** output,Type  *** AllInput)
{
   mInPut = AllInput[0];
   // ShowLine(mInPut[0],aW2Digeo);
   if (this->first_line())
   {
      for (INT y=this->y0Buf(); y<this->y1Buf()-1 ; y++)
           ConvolLine(y);
   }
   ConvolLine(this->y1Buf()-1);

   // ShowLine(mLineFiltered[0],aW3Digeo);

  
  mConvol->ConvolCol(output[0],mLineFiltered,this->x0(),this->x1(),0);
  rotate_plus_data(mLineFiltered,this->y0Buf(),this->y1Buf());

}

Fonc_Num LinearSepFilter
         (
             Fonc_Num                aFonc,
             cConvolSpec<INT> *    aFiltrI,
             cConvolSpec<double> * aFiltrD
         )
{
  ELISE_ASSERT(aFiltrI->Sym() && aFiltrD->Sym(),"LinearSepFilter handle only symetric filter");

  bool IntF = aFonc.integral_fonc(true);
  int aD = IntF ? aFiltrI->Fin() :  aFiltrD->Fin() ;

  

  //if (aD != aFiltrD->Fin())
  //{
  //    std::cout << "FILTRE, SzI " << aFiltrI->Fin() << " " << aFiltrD->Fin() << "\n";
  //    ELISE_ASSERT(false,"Incoh INT/DOUBLE in LinearSepFilter");
  //}

  return create_op_buf_simple_tpl
            (

                IntF ? new cSomFiltreSep<INT>(aFiltrI) : 0,
                IntF ? 0 : new cSomFiltreSep<double>(aFiltrD),
                aFonc,
                1,
                Box2di(aD)
            );

}

*/

/****************************************/
/*                                      */
/*           cConvolSpec                */
/*                                      */
/****************************************/

template <class tData> 
cConvolSpec<tData>::cConvolSpec( const TBASE *aFilter, int aDeb, int aFin, int aNbShift ){ set( aFilter, aDeb, aFin, aNbShift ); }

template <class tData> 
cConvolSpec<tData>::cConvolSpec( const ConvolutionKernel1D<TBASE> &aKernel ){ set( aKernel.data(), aKernel.begin(), aKernel.end(), aKernel.nbShift() ); }

template <class tData> 
void cConvolSpec<tData>::set( const TBASE *aFilter, int aDeb, int aFin, int aNbShift )
{
	mNbShift = aNbShift;
	mDeb = aDeb;
	mFin = aFin;
	mSym = true;

	mCoeffs.resize( mFin-mDeb+1 );
	memcpy( mCoeffs.data(), aFilter+mDeb, mCoeffs.size()*sizeof(TBASE) );
	mDataCoeff = mCoeffs.data()-mDeb;

	if ( aFin+aDeb!=0 )
		mSym = false;
	else
	{
		for ( int aK=1; aK<=mFin; aK++ )
			if ( Abs(mDataCoeff[aK]-mDataCoeff[-aK])>EPSILON_SYMMETRY ) mSym = false;
	}
}

template <class tData> 
cConvolSpec<tData>::cConvolSpec( const cConvolSpec<tData> &i_b ){ set(i_b); }

template <class tData>
void cConvolSpec<tData>::set( const cConvolSpec<tData> &i_b )
{
	mNbShift = i_b.mNbShift;
	mDeb = i_b.mDeb;
	mFin = i_b.mFin;
	mCoeffs = i_b.mCoeffs;
	mDataCoeff = mCoeffs.data()-mDeb;
	mSym = i_b.mSym;
}

template <class tData>
cConvolSpec<tData> & cConvolSpec<tData>::operator =( const cConvolSpec<tData> &i_b )
{
	set(i_b);
	return *this;
}

template <class tData>
bool cConvolSpec<tData>::Match( const ConvolutionKernel1D<TBASE> &aKernel ) const
{
	if ( (aKernel.nbShift()!=(unsigned int)mNbShift) || (aKernel.begin()!=mDeb) || (aKernel.end()!=mFin) ) return false;

	const TBASE *data = aKernel.data();
	for ( int aK=mDeb; aK<=mFin; aK++ )
		if ( Abs( mDataCoeff[aK]-data[aK] )>EPSILON_MATCH ) return false;

	return true;
}

template <class tData>
cConvolSpec<tData> * cConvolSpec<tData>::duplicate() const { return new cConvolSpec<tData>(*this); }

template <class tData>
cConvolSpec<tData>::~cConvolSpec(){}

template <class tData>
void cConvolSpec<tData>::ConvolCol(tData * Out,tData **In,int aX0,int aX1,int anYIn) const
{
    tData aV0 = mDataCoeff[0];
    tData * aL0 = In[anYIn];
    for (int anX = aX0; anX<aX1 ; anX++)
    {
          Out[anX] = aL0[anX] * aV0;
    }


    for (int aDY= (mSym?1:mDeb)  ; aDY<=mFin ; aDY++)
    {
        if (aDY)
        {
            tData *aLP = In[anYIn+aDY];
            tData  aVP = mDataCoeff[aDY];
            if (mSym)
            {
                tData *aLM = In[anYIn-aDY];
                for (int anX = aX0; anX<aX1 ; anX++)
                {
                   Out[anX] += aVP * (aLP[anX] + aLM[anX]) ;
                }
            }
            else
            {
                for (int anX = aX0; anX<aX1 ; anX++)
                {
                   Out[anX] += aVP * aLP[anX];
                }
            }
        }
    }
    for (int anX = aX0; anX<aX1 ; anX++)
    {
          Out[anX] = ShiftDr(Out[anX],mNbShift);
    }
}


template <class tData>
void cConvolSpec<tData>::Convol( tData *Out, const tData *In, int aK0, int aK1 ) const
{
   In += aK0;
   if (mSym)
   {
      for (int aK= aK0 ; aK<aK1 ; aK++)
      {
          TBASE aRes = In[0] * mDataCoeff[0];
          for (int aDelta=1 ; aDelta <= mFin ; aDelta++)
          {
              aRes +=  ((TBASE)(In[aDelta])+(TBASE)(In[-aDelta])) * mDataCoeff[aDelta];
          }
          Out[aK] = ShiftDr(aRes,mNbShift);
          In++;
      }
   }
   else
   {
      for (int aK= aK0 ; aK<aK1 ; aK++)
      {
          TBASE aRes = 0;
          for (int aDelta=mDeb ; aDelta <= mFin ; aDelta++)
          {
              aRes +=  In[aDelta] * mDataCoeff[aDelta];
          }
          Out[aK] = ShiftDr(aRes,mNbShift);
          In++;
      }
   }
}

template <class tData>        bool          cConvolSpec<tData>::IsCompiled() const { return false; }
template <class tData> inline int           cConvolSpec<tData>::Deb()        const { return mDeb; }
template <class tData> inline int           cConvolSpec<tData>::Fin()        const { return mFin; }
template <class tData> inline bool          cConvolSpec<tData>::Sym()        const { return mSym; }
template <class tData> inline int           cConvolSpec<tData>::NbShift()    const { return mNbShift; }
template <class tData> inline const TBASE * cConvolSpec<tData>::DataCoeff()  const { return mDataCoeff; }


//----------------------------------------------------------------------
// ConvolutionHandler methods
//----------------------------------------------------------------------

// include compiled kernel
#include "GenConvolSpec.u_int1.h"
#include "GenConvolSpec.u_int2.h"
#include "GenConvolSpec.real4.h"

#ifndef HAS_U_INT1_COMPILED_CONVOLUTIONS
template <> void ConvolutionHandler<U_INT1>::addCompiledKernels(){}
#endif

#ifndef HAS_U_INT2_COMPILED_CONVOLUTIONS
template <> void ConvolutionHandler<U_INT2>::addCompiledKernels(){}
#endif

#ifndef HAS_REAL4_COMPILED_CONVOLUTIONS
template <> void ConvolutionHandler<REAL4>::addCompiledKernels(){}
#endif

template <class tData>
ConvolutionHandler<tData>::ConvolutionHandler(){ addCompiledKernels(); }

template <class tData>
void ConvolutionHandler<tData>::clear()
{
	typename list<cConvolSpec<tData> *>::iterator it = mConvolutions.begin();
	while ( it!=mConvolutions.end() )
		delete *it++;
	mConvolutions.clear();
}

template <class tData>
void ConvolutionHandler<tData>::set( const ConvolutionHandler<tData> &i_b )
{
	clear();
	// add not-compiled kernels
	typename list<cConvolSpec<tData>*>::const_iterator it = i_b.mConvolutions.begin();
	while ( it!=i_b.mConvolutions.end() )
		mConvolutions.push_back( (**it++).duplicate() );
}

template <class tData>
ConvolutionHandler<tData>::ConvolutionHandler( const ConvolutionHandler<tData> &aSrc ){ set(aSrc); }

template <class tData>
ConvolutionHandler<tData> & ConvolutionHandler<tData>::operator =( const ConvolutionHandler<tData> &aSrc )
{
	set(aSrc);
	return *this;
}

template <class tData>
ConvolutionHandler<tData>::~ConvolutionHandler(){ clear(); }

std::string NameClassConvSpec_( const string i_typename, unsigned int i_iConvolution )
{
	stringstream ss;
	ss << "cConvolSpec_" << i_typename << "_Num" << i_iConvolution;
	return ss.str();
}

static void LineSym( ostream &aFile,int aVal,int aK)   { aFile << "\t\t\t\t              +   " << aVal << "*(INT(In[" << aK << "])+INT(In[" << -aK << "]))" << endl; }
static void LineSym( ostream &aFile,double aVal,int aK){ aFile << "\t\t\t\t              +   " << aVal << "*(REAL(In[" << aK << "])+REAL(In[" << -aK << "]))" << endl; }

static void LineStd( ostream &aFile,int aVal,int aK){ aFile << "\t\t\t\t              +   " << aVal << "*(In[" << aK << "])" << endl; }
static void LineStd( ostream &aFile,double aVal,int aK){ aFile << "\t\t\t\t              +   " << aVal << "*(In[" << aK << "])" << endl; }

static void  PutVal( ostream &aFile,int aVal){ aFile << aVal; }
static void  PutVal( ostream &aFile,double aVal){ aFile << aVal; }

template <class tData> 
bool ConvolutionHandler<tData>::generateCode( const string &i_filename ) const
{
	ofstream f( i_filename.c_str() );
	if ( !f ) return false;

	std::string aNType = TypeTraits<tData>::Name();
	std::string aNTBase = TypeTraits<TBASE>::Name();

	// generate classes
	typename list<cConvolSpec<tData> *>::const_iterator itConvolution = mConvolutions.begin();
	unsigned int iConvolution = 0;
	while ( itConvolution!=mConvolutions.end() )
	{
		cConvolSpec<tData> &convolution = **itConvolution++;
		std::string aNClass = NameClassConvSpec_( aNType, iConvolution++ );
		const int aFin  = convolution.Fin();
		const int aDeb  = convolution.Deb();
		const int aNbShift = convolution.NbShift();
		const TBASE *aFilter = convolution.DataCoeff();

		f << "#define HAS_" << aNType << "_COMPILED_CONVOLUTIONS " << endl;
		f << endl;
		f << "class " << aNClass << " : public cConvolSpec<" << aNType << ">" << endl;
		f << "{" << endl;
		f << "\tpublic :" << endl;
		f << "\t\tbool IsCompiled() const { return true; }" << endl;
		f << "\t\tcConvolSpec<" << aNType << "> * duplicate() const { return new " << aNClass << "(*this); }" << endl;
		f << "\t\tvoid Convol(" << aNType << " *Out, const " << aNType << " * In,int aK0,int aK1) const" << endl;
		f << "\t\t{" << endl;
		f << "\t\t\tIn+=aK0;" << endl;
		f << "\t\t\tOut+=aK0;" << endl;
		f << "\t\t\tfor (int aK=aK0; aK<aK1 ; aK++){" << endl;
		f << "\t\t\t\t*(Out++) =  (" << endl;
		/*
		if ( numeric_limits<tData>::is_integer )
			f << "\t\t\t\t                  " << (1<<aNbShift)/2 << endl;
		else
			f << "\t\t\t\t                  0" << endl;
		*/
		for (int aK=aDeb ; aK <=aFin ; aK++)
			if ((-aK>=aDeb) && (-aK<=aFin) && (aK) && (aFilter[aK]==aFilter[-aK]))
			{
				if (aK<0) LineSym( f, aFilter[aK], aK );
			}
			else
				LineStd( f, aFilter[aK], aK );
		if ( numeric_limits<tData>::is_integer )
			f << "\t\t\t\t            )>>" << aNbShift << ";" << endl;
		else
			f << "                           );" << endl;
		f << "\t\t\t\tIn++;" << endl;
		f << "\t\t\t}" << endl;
		f << "\t\t}\n" << endl;
		f << "\t\t" << aNClass << "(" << aNTBase << " * aFilter):";
		f << "cConvolSpec<" << aNType << ">(aFilter-(" << aDeb << ")," << aDeb << "," << aFin << "," << aNbShift << "){}" << endl;
		f << "};\n" << endl;
	}

	f << "template <> void ConvolutionHandler<" << aNType << ">::addCompiledKernels()" << endl;
	f << "{" << endl;
	itConvolution = mConvolutions.begin();
	iConvolution = 0;
	while ( itConvolution!=mConvolutions.end() )
	{
		cConvolSpec<tData> &convolution = **itConvolution++;

		std::string aNClass = NameClassConvSpec_( aNType, iConvolution++ );
		const int aFin  = convolution.Fin();
		const int aDeb  = convolution.Deb();
		const TBASE *aFilter = convolution.DataCoeff();

		f << "\t{" << endl;
		f << "\t\t" << aNTBase << " theCoeff[" << aFin-aDeb+1 << "] = {";
		for (int aK=aDeb ; aK <=aFin ; aK++)
		{
			if (aK!=aDeb) f << ",";
			PutVal( f, aFilter[aK] );
		}
		f << "};" << endl;
		f << "\t\tmConvolutions.push_back( new " << aNClass << "(theCoeff) );" << endl;
		f << "\t}" << endl;
	}
	f << "}" << endl;

	return true;
}

template <class tData>
cConvolSpec<tData> * ConvolutionHandler<tData>::getExistingConvolution( const ConvolutionKernel1D<TBASE> &aKernel )
{
	typename list<cConvolSpec<tData> *>::iterator itKernel = mConvolutions.begin();
	while ( itKernel!=mConvolutions.end() )
	{
		if ( (**itKernel).Match(aKernel) ) return *itKernel;
		itKernel++;
	}
	return NULL;
}

template <class tData>
cConvolSpec<tData> * ConvolutionHandler<tData>::getConvolution( const ConvolutionKernel1D<TBASE> &aKernel )
{
	cConvolSpec<tData> *aRes = getExistingConvolution(aKernel);
	if ( aRes!=NULL ) return aRes;

	mConvolutions.push_back( new cConvolSpec<tData>(aKernel) );
	return mConvolutions.back();
}

template <class tData>
size_t ConvolutionHandler<tData>::nbConvolutionsNotCompiled() const
{
	size_t res = 0;
	typename list<cConvolSpec<tData> *>::const_iterator it = mConvolutions.begin();
	while ( it!=mConvolutions.end() )
		if ( !(**it++).IsCompiled() ) res++;
	return res;
}

template <class tData>
size_t ConvolutionHandler<tData>::nbConvolutions() const { return mConvolutions.size(); }

template <class tData>
std::string ConvolutionHandler<tData>::defaultCodeBasename()
{
	string lowTypeName = TypeTraits<tData>::Name();
	for ( size_t i=0; i<lowTypeName.length(); i++ )
		lowTypeName[i] = tolower( lowTypeName[i] );
	return string("GenConvolSpec.")+lowTypeName+".h";
}

//----------------------------------------------------------------------
// include compiled kernels
//----------------------------------------------------------------------

const int PackTranspo = 4;

/*
Im1D_REAL8 MakeSom1( Im1D_REAL8 &i_vector );

//  K3-C3 = K1-C1 + K2-C2

Im1D_REAL8 DeConvol(int aC2,int aSz2,Im1D_REAL8 aI1,int aC1,Im1D_REAL8 aI3,int aC3)
{
   L2SysSurResol aSys(aSz2);
   aSys.SetPhaseEquation(0);
   
   int  aSz1 = aI1.tx();
   int  aSz3 = aI3.tx();

   for (int aK3 =0 ; aK3 < aSz3 ; aK3++)
   {
       std::vector<int> aVInd;
       std::vector<double> aVCoef;
       for (int aK=0; aK < aSz1 ; aK++)
       {
           int aK1 = aK;
           int aK2 = aC2 + (aK3-aC3) - (aK1-aC1);
           if ((aK1>=0)&&(aK1<aSz1)&&(aK2>=0)&&(aK2<aSz2))
           {
               aVInd.push_back(aK2);
               aVCoef.push_back(aI1.data()[aK1]);
           }
       }
       if (aVInd.size()) 
       {
          aSys.GSSR_AddNewEquation_Indexe
          (
                0,0,0,
                aVInd,
                1.0,
                &(aVCoef.data()[0]),
                aI3.data()[aK3]
          );
       }
   }

   Im1D_REAL8 aRes = aSys.GSSR_Solve(0);
   ELISE_COPY(aRes.all_pts(),Max(aRes.in(),0.0),aRes.out());
   return MakeSom1(aRes);
}

Im1D_REAL8 DeConvol(int aDemISz2,Im1D_REAL8 aI1,Im1D_REAL8 aI3)
{
   ELISE_ASSERT((aI1.tx()%2)&&(aI3.tx()%2),"Parity error in DeConvol");
   return DeConvol(aDemISz2,1+2*aDemISz2,aI1,aI1.tx()/2,aI3,aI3.tx()/2);
}

Im1D_REAL8 Convol(Im1D_REAL8 aI1,int aC1,Im1D_REAL8 aI2,int aC2)
{
    Im1D_REAL8 aRes(aI1.tx()+aI2.tx()-1,0.0);

    ELISE_COPY
    (
         rectangle(Pt2di(0,0),Pt2di(aRes.tx(),aRes.tx())),
         aI1.in(0)[FX]*aI2.in(0)[FY-FX],
         aRes.histo(true).chc(FY)
    );

   return aRes;
}

Im1D_REAL8 Convol(Im1D_REAL8 aI1,Im1D_REAL8 aI2)
{
   ELISE_ASSERT((aI1.tx()%2)&&(aI2.tx()%2),"Parity error in Convol");
   return Convol(aI1,aI1.tx()/2,aI2,aI2.tx()/2);
}
*/

template <class T>
inline void setMax( T &v1, T v2 ){ if (v1<v2) v1=v2; }

template <class T>
inline void setMin( T &v1, T v2 ){ if (v1>v2) v1=v2; }

// Pour utiliser un filtre sur les bord, clip les intervalle
// pour ne pas deborder et renvoie la somme partielle
template <class tBase> tBase ClipForConvol( int aSz, int aKXY, const tBase *aData, int & aDeb, int &aFin )
{
	setMax(aDeb,-aKXY);
	setMin(aFin,aSz-1-aKXY);

	tBase aSom = 0;
	for (int aK= aDeb ; aK<=aFin ; aK++)
		aSom += aData[aK];

	return aSom;
}



   // Produit scalaire basique d'un filtre lineaire avec une ligne
   // et une colonne image
template <class Type,class tBase> 
inline tBase CorrelLine(tBase aSom,const Type * aData1,const tBase *  aData2,const int & aDeb,const int & aFin)
{


     for (int aK= aDeb ; aK<=aFin ; aK++)
        aSom += aData1[aK]*aData2[aK];

   return aSom;
}

inline int getNbShift( const REAL *i_data, size_t i_nbElements ){ return 0; }

inline int getNbShift( const INT *i_data, size_t i_nbElements )
{
	INT sum = 0;
	while ( i_nbElements-- ) sum += *i_data++;
	int nbShift = (int)( log((double)sum)/log(2.)+0.5 );

	ELISE_DEBUG_ERROR( sum!=(1<<nbShift), "getNbShift", "sum = " << sum << " != 2^" << nbShift );

	return nbShift;
}

// anX must not be lesser than 0
template <class tData> 
void SetConvolBordX
	(
		const tData **i_srcData, const int i_width, const int i_height,
		tData **o_dstData,
		int anX,
		const TBASE *aDFilter, int aDebX, int aFinX
	)
{
	TBASE aDiv = ClipForConvol<TBASE>( i_width, anX, aDFilter, aDebX, aFinX );
	const TBASE aSom = InitFromDiv( aDiv, (TBASE*)0 );

	for (int anY=0 ; anY<i_height ; anY++)
		o_dstData[anY][anX] = CorrelLine( aSom, i_srcData[anY]+anX, aDFilter, aDebX, aFinX )/aDiv;
}


    //  SetConvolSepX(aImIn,aData,-aSzKer,aSzKer,aNbShitXY,aCS);
template <class tData> 
void SetConvolSepX
	(
		const tData **i_srcData, const int i_width, const int i_height,
		tData **i_dstData,
		const cConvolSpec<tData> &aConvolution1d
	)
{
	int aX0 = std::min( -aConvolution1d.Deb(), i_width );

	int anX;
	for (anX=0; anX <aX0 ; anX++)
		SetConvolBordX( i_srcData, i_width, i_height, i_dstData, anX, aConvolution1d.DataCoeff(), aConvolution1d.Deb(), aConvolution1d.Fin() );

	int aX1 = std::max( i_width-aConvolution1d.Fin(), anX );
	for ( anX=aX1; anX<i_width; anX++ ) // max car aX1 peut être < aX0 voir negatif et faire planter
		SetConvolBordX( i_srcData, i_width, i_height, i_dstData, anX, aConvolution1d.DataCoeff(), aConvolution1d.Deb(), aConvolution1d.Fin() );

	// const tBase aSom = InitFromDiv(ShiftG(tBase(1),aNbShitX),(tBase*)0);
	for (int anY=0 ; anY<i_height ; anY++)
	{
		tData *aDOut = i_dstData[anY];
		const tData *aDIn =  i_srcData[anY];

		aConvolution1d.Convol(aDOut,aDIn,aX0,aX1);
	}
}

template <class tData> 
tData ** new_data_lines( int i_width, int i_height )
{
	tData *data = new tData[i_width*i_height];
	tData **lines = new tData*[i_height];
	for ( int y=0; y<i_height; y++ )
	{
		lines[y] = data;
		data += i_width;
	}
	return lines;
}

// i_data must have at least one line
template <class tData> 
void delete_data_lines( tData **i_data )
{
	delete [] i_data[0];
	delete [] i_data;
}

template <class tData> 
void SelfSetConvolSepY
	(
		tData **i_data, int i_width, int i_height,
		const cConvolSpec<tData> &aConvolution1d
	)
{
	tData **aBufIn  = new_data_lines<tData>(i_height, PackTranspo);
	tData **aBufOut = new_data_lines<tData>(i_height, PackTranspo);
	for (int anX = 0; anX < i_width; anX += PackTranspo)
	{
		// Il n'y a pas de debordement car les images  sont predementionnee 
		// d'un Rab de PackTranspo;  voir ResizeBasic

		tData * aL0 = aBufIn[0];
		tData * aL1 = aBufIn[1];
		tData * aL2 = aBufIn[2];
		tData * aL3 = aBufIn[3];
		for (int aY = 0 ; aY < i_height; aY++)
		{
			const tData * aL = i_data[aY] + anX;
			*aL0++ = *aL++;
			*aL1++ = *aL++;
			*aL2++ = *aL++;
			*aL3++ = *aL++;
		}
		SetConvolSepX( (const tData**)aBufIn, i_height, PackTranspo, aBufOut, aConvolution1d );

		aL0 = aBufOut[0];
		aL1 = aBufOut[1];
		aL2 = aBufOut[2];
		aL3 = aBufOut[3];

		for (int aY = 0; aY < i_height; aY++)
		{
			tData * aL = i_data[aY] + anX;
			*aL++ = *aL0++;
			*aL++ = *aL1++;
			*aL++ = *aL2++;
			*aL++ = *aL3++;
		}
	}

	delete_data_lines(aBufIn);
	delete_data_lines(aBufOut);
}

template <class tData>
void convolution( const tData **aSrcData, const int aWidth, const int aHeight, const cConvolSpec<tData> &aConvolution, tData **aDstData )
{
	SetConvolSepX( aSrcData, aWidth, aHeight, aDstData, aConvolution );
	SelfSetConvolSepY( aDstData, aWidth, aHeight, aConvolution );
}


//----------------------------------------------------------------------
// functions of legacy_convolution
//----------------------------------------------------------------------

template <class tData>
void legacy_convolution_transpose( const tData *i_src, const int i_width, const int i_height, const vector<TBASE> &i_kernel, int i_nbShift, tData *o_dst )
{
	ELISE_DEBUG_ERROR( i_kernel.size()%2==0, "LegacyConvolution_transpose<" << TypeTraits<tData>::Name() << ">", "i_kernel.size()%2==0" );

	// convolve along columns, save transpose
	// filter is (2*W+1) by 1
	const int W = (int)((i_kernel.size() - 1) / 2);
	const TBASE *filter_pt = i_kernel.data();
	const tData *src_pt = i_src;
	for ( int j=0; j<i_height; ++j )
	{
		for ( int i=0; i<i_width; ++i )
		{
			TBASE acc = 0, x;
			const TBASE *g = filter_pt;
			const tData *start = src_pt+( i-W ), *stop;

			// beginning
			stop = src_pt ;
			x = *stop ;
			while ( start<=stop )
			{
				acc += (*g++)*x;
				start++;
			}

			// middle
			stop =  src_pt + std::min(i_width-1, i+W) ;
			while ( start<stop )
				acc += (*g++)*(*start++);

			// end
			x  = *start ;
			stop = src_pt + (i+W);
			while( start<=stop ) { acc += (*g++)*x; start++; }

			// save
			*o_dst = (tData)ShiftDr(acc,i_nbShift);
			o_dst += i_width;
		}
		// next column
		src_pt += i_width;
		o_dst -= i_width*i_height-1;
	}
}

template <class tData>
void legacy_convolution( const tData *aSrcData, const int aWidth, const int aHeight, tData *aTmpData, const ConvolutionKernel1D<TBASE> &aKernel, tData *aDstData )
{
	legacy_convolution_transpose( aSrcData, aWidth, aHeight, aKernel.coefficients(), aKernel.nbShift(), aTmpData );
	legacy_convolution_transpose( aTmpData, aWidth, aHeight, aKernel.coefficients(), aKernel.nbShift(), aDstData );
}


//----------------------------------------------------------------------
// instantiation
//----------------------------------------------------------------------

template class cConvolSpec<U_INT1>;
template class ConvolutionHandler<U_INT1>;
template void convolution<U_INT1>( const U_INT1 **aSrcData, const int aWidth, const int aHeight, const cConvolSpec<U_INT1> &aConvolution, U_INT1 **aDstData );
template void legacy_convolution<U_INT1>( const U_INT1 *aSrcData, const int aWidth, const int aHeight, U_INT1 *aTmpData, const ConvolutionKernel1D<INT> &aKernel, U_INT1 *aDstData );

template class cConvolSpec<U_INT2>;
template class ConvolutionHandler<U_INT2>;
template void convolution<U_INT2>( const U_INT2 **aSrcData, const int aWidth, const int aHeight, const cConvolSpec<U_INT2> &aConvolution, U_INT2 **aDstData );
template void legacy_convolution<U_INT2>( const U_INT2 *aSrcData, const int aWidth, const int aHeight, U_INT2 *aTmpData, const ConvolutionKernel1D<INT> &aKernel, U_INT2 *aDstData );

template class cConvolSpec<REAL4>;
template class ConvolutionHandler<REAL4>;
template void convolution<REAL4>( const REAL4 **aSrcData, const int aWidth, const int aHeight, const cConvolSpec<REAL4> &aConvolution, REAL4 **aDstData );
template void legacy_convolution<REAL4>( const REAL4 *aSrcData, const int aWidth, const int aHeight, REAL4 *aTmpData, const ConvolutionKernel1D<REAL> &aKernel, REAL4 *aDstData );

template U_INT1 ** new_data_lines<U_INT1>( int i_width, int i_height );
template U_INT2 ** new_data_lines<U_INT2>( int i_width, int i_height );
template U_INT4 ** new_data_lines<U_INT4>( int i_width, int i_height );
template U_INT8 ** new_data_lines<U_INT8>( int i_width, int i_height );
template INT1 ** new_data_lines<INT1>( int i_width, int i_height );
template INT2 ** new_data_lines<INT2>( int i_width, int i_height );
template INT4 ** new_data_lines<INT4>( int i_width, int i_height );
template _INT8 ** new_data_lines<_INT8>( int i_width, int i_height );
template REAL4 ** new_data_lines<REAL4>( int i_width, int i_height );
template REAL8 ** new_data_lines<REAL8>( int i_width, int i_height );
template REAL16 ** new_data_lines<REAL16>( int i_width, int i_height );

template void delete_data_lines<U_INT1>( U_INT1 **i_data );
template void delete_data_lines<U_INT2>( U_INT2 **i_data );
template void delete_data_lines<U_INT4>( U_INT4 **i_data );
template void delete_data_lines<U_INT8>( U_INT8 **i_data );
template void delete_data_lines<INT1>( INT1 **i_data );
template void delete_data_lines<INT2>( INT2 **i_data );
template void delete_data_lines<INT4>( INT4 **i_data );
template void delete_data_lines<_INT8>( _INT8 **i_data );
template void delete_data_lines<REAL4>( REAL4 **i_data );
template void delete_data_lines<REAL8>( REAL8 **i_data );
template void delete_data_lines<REAL16>( REAL16 **i_data );

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
