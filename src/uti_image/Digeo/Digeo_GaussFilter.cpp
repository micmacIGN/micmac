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

//#include "Digeo.h"

// Test

/****************************************/
/*                                      */
/*             ::                       */
/*                                      */
/****************************************/

Im1D_REAL8 MakeSom( Im1D_REAL8 &i_vector, double i_dstSum )
{
	double aSomActuelle;
	Im1D_REAL8 aRes( i_vector.tx() );
	ELISE_COPY( i_vector.all_pts(), i_vector.in(), sigma(aSomActuelle) );
	ELISE_COPY( i_vector.all_pts(), i_vector.in()*( i_dstSum/aSomActuelle ), aRes.out() );
	return aRes;
}

Im1D_REAL8 MakeSom1( Im1D_REAL8 &i_vector )
{
	return MakeSom(i_vector,1.);
}

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
                aI3.data()[aK3],
                NullPCVU
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

   // Pour utiliser un filtre sur les bord, clip les intervalle
   // pour ne pas deborder et renvoie la somme partielle
template <class tBase>
tBase ClipForConvol( int aSz, int aKXY, const tBase * aData, int & aDeb, int &aFin )
{
    ElSetMax(aDeb,-aKXY);
    ElSetMin(aFin,aSz-1-aKXY);

    tBase aSom = 0;
    for (int aK= aDeb ; aK<=aFin ; aK++)
        aSom += aData[aK];

   return aSom;
}



   // Produit scalaire basique d'un filtre lineaire avec une ligne
   // et une colonne image
template <class Type,class tBase> 
inline tBase CorrelLine( tBase aSom, const Type *aData1, const tBase *aData2, const int & aDeb, const int &aFin )
{


     for (int aK= aDeb ; aK<=aFin ; aK++)
        aSom += aData1[aK]*aData2[aK];

   return aSom;
}

template <class Type,class tBase>
cConvolSpec<Type> * ToCompKer( const tBase *aKernel, int aKernelSize, int aNbShift )
{
	ELISE_ASSERT( aKernelSize%2, "ToCompKer: kernel size should be odd" );
	aKernelSize /= 2;

	const tBase * aData = aKernel+aKernelSize;
	while ( aKernelSize && (aData[aKernelSize]==0) && (aData[-aKernelSize]==0) )
		aKernelSize--;

	return cConvolSpec<Type>::GetOrCreate( aData, -aKernelSize, aKernelSize, aNbShift, false ) ;
}


/****************************************/
/*                                      */
/*             cTplImInMem              */
/*                                      */
/****************************************/

inline REAL InitFromDiv(REAL,REAL *) { return 0; }
inline INT  InitFromDiv(INT aDiv,INT *) { return aDiv/2; }

// anX must not be lesser than 0
template <class Type> 
void  cTplImInMem<Type>::SetConvolBordX
      (
          Im2D<Type,tBase> aImOut,
          Im2D<Type,tBase> aImIn,
          int anX,
          const tBase * aDFilter,int aDebX,int aFinX
      )
{
    tBase aDiv = ClipForConvol(aImOut.tx(),anX,aDFilter,aDebX,aFinX);
    Type ** aDOut = aImOut.data();
    Type ** aDIn = aImIn.data();

    const tBase aSom = InitFromDiv(aDiv,(tBase*)0);

    int aSzY = aImOut.ty();
    for (int anY=0 ; anY<aSzY ; anY++)
        aDOut[anY][anX] = CorrelLine(aSom,aDIn[anY]+anX,aDFilter,aDebX,aFinX) / aDiv;
}


    //  SetConvolSepX(aImIn,aData,-aSzKer,aSzKer,aNbShitXY,aCS);
template <class Type> 
void cTplImInMem<Type>::SetConvolSepX
     (
          Im2D<Type,tBase> aImOut,
          Im2D<Type,tBase> aImIn,
          int  aNbShitX,
          cConvolSpec<Type> * aCS
     )
{
    ELISE_ASSERT(aImOut.sz()==aImIn.sz(),"Sz in SetConvolSepX");
    
    // __DEL
    Im2D<Type,tBase> src = aImIn;
    if ( aImOut.data_lin()==aImIn.data_lin() )
    {
		 Im2D<Type,tBase> newSrc( aImIn.tx()+PackTranspo, aImIn.ty() );
		 newSrc.Resize( aImIn.sz() );
		 memcpy( newSrc.data_lin(), aImIn.data_lin(), aImIn.tx()*aImIn.ty()*sizeof(Type) );
		 aImIn = newSrc;
	 }

    int aSzX = aImOut.tx();
    int aSzY = aImOut.ty();
    int aX0 = std::min( -aCS->Deb(), aSzX );

    int anX;
    for (anX=0; anX <aX0 ; anX++)
        SetConvolBordX(aImOut,aImIn,anX,aCS->DataCoeff(),aCS->Deb(),aCS->Fin());

    int aX1 = std::max( aSzX-aCS->Fin(), anX );
    for (anX =aX1; anX <aSzX ; anX++) // max car aX1 peut être < aX0 voir negatif et faire planter
        SetConvolBordX(aImOut,aImIn,anX,aCS->DataCoeff(),aCS->Deb(),aCS->Fin());
   
    // const tBase aSom = InitFromDiv(ShiftG(tBase(1),aNbShitX),(tBase*)0);
    for (int anY=0 ; anY<aSzY ; anY++)
    {
        Type * aDOut = aImOut.data()[anY];
        Type * aDIn =  aImIn.data()[anY];

        aCS->Convol(aDOut,aDIn,aX0,aX1);
    }
}


template <class Type> 
void cTplImInMem<Type>::SetConvolSepX
     (
          const cTplImInMem<Type> & aImIn,
          int  aNbShitX,
          cConvolSpec<Type> * aCS
     )
{
      SetConvolSepX
      (
         mIm,aImIn.mIm, 
         aNbShitX,
         aCS
      );
}


template <class Type> 
void cTplImInMem<Type>::SelfSetConvolSepY
     (
          int  aNbShitY,
          cConvolSpec<Type> * aCS
     )
{
    Im2D<Type,tBase> aBufIn(mSz.y,PackTranspo);
    Im2D<Type,tBase> aBufOut(mSz.y,PackTranspo);

    Type ** aData =  mIm.data();

	#ifdef __DEBUG_DIGEO
		for ( int y=0; y<mIm.ty(); y++ )
		{
			if ( aData[y]!=mIm.data_lin()+y*mIm.tx() )
			{
				cerr << "cTplImInMem: check line failed for line " << y << endl;
				exit(-1);
			}
		}
		if ( mIm.linearDataAllocatedSize()<(mIm.tx()+PackTranspo)*mIm.ty() )
		{
			cerr << "cTplImInMem: linearDataAllocatedSize = " << mIm.linearDataAllocatedSize() << " < (i_image.tx()+PackTranspo)*i_image.ty() = " << (mIm.tx()+PackTranspo)*mIm.ty() << endl;
			exit(-1);
		}
	#endif

    for (int anX = 0; anX<mSz.x ; anX+=PackTranspo)
    {
         // Il n'y a pas de debordement car les images  sont predementionnee 
         // d'un Rab de PackTranspo;  voir ResizeBasic

         Type * aL0 = aBufIn.data()[0];
         Type * aL1 = aBufIn.data()[1];
         Type * aL2 = aBufIn.data()[2];
         Type * aL3 = aBufIn.data()[3];
         for (int aY=0 ; aY<mSz.y ; aY++)
         {
             Type * aL = aData[aY]+anX;
             *(aL0)++ = *(aL++);
             *(aL1)++ = *(aL++);
             *(aL2)++ = *(aL++);
             *(aL3)++ = *(aL++);
         }
         SetConvolSepX(aBufOut,aBufIn,aNbShitY,aCS);

         aL0 = aBufOut.data()[0];
         aL1 = aBufOut.data()[1];
         aL2 = aBufOut.data()[2];
         aL3 = aBufOut.data()[3];

         for (int aY=0 ; aY<mSz.y ; aY++)
         {
             Type * aL = aData[aY]+anX;
             *(aL)++ = *(aL0++);
             *(aL)++ = *(aL1++);
             *(aL)++ = *(aL2++);
             *(aL)++ = *(aL3++);
         }
    }
}

/*
template <class Type> 
void cTplImInMem<Type>::SetConvolSepXY
     (
          bool Increm,
          double aSigma,
          const cTplImInMem<Type> & aImIn,
          Im1D<tBase,tBase> aKerXY,
          int  aNbShitXY
     )
{
	ELISE_ASSERT(mSz==aImIn.mSz,"Size im diff in ::SetConvolSepXY");

	mAppli.times()->start();

	cConvolSpec<Type> * aCS=  ToCompKer<Type,tBase>( aKerXY, aNbShitXY, aSigma, Increm );

	if ( !aCS->IsCompiled() ) mAppli.upNbSlowConvolutionsUsed<Type>();

	SetConvolSepX(aImIn,aNbShitXY,aCS);
	SelfSetConvolSepY(aNbShitXY,aCS);

	mAppli.times()->stop("gaussian convolution");
}
*/

/*
void TestConvol()
{
   int aT1 = 5;
   int aT2 = 3;
   Im1D_REAL8  aI1 = DigeoGaussianKernel(2.0,aT1,10);
   Im1D_REAL8  aI2 = DigeoGaussianKernel(1.0,aT2,10);

   // ELISE_COPY(aI1.all_pts(),FX==aT1,aI1.out());
   // ELISE_COPY(aI2.all_pts(),FX==aT2,aI2.out());

   Im1D_REAL8  aI3 = Convol(aI1,aI2);

   Im1D_REAL8 aI2B = DeConvol(2,aI1,aI3);

   Im1D_REAL8 aI4 = Convol(aI1,aI2B);

   for (int aK=0 ; aK<ElMax(aI3.tx(),aI4.tx()) ; aK++)
   {
       std::cout 
                  << aK << ":" 
                 << "  " << (aK<aI3.tx() ? ToString(aI3.data()[aK]) : " XXXXXX " )
                 << "  " << (aK<aI1.tx() ? ToString(aI1.data()[aK]) : " XXXXXX " )
                 << "  " << (aK<aI2.tx() ? ToString(aI2.data()[aK]) : " XXXXXX " )
                 << "  " << (aK<aI2B.tx() ? ToString(aI2B.data()[aK]) : " XXXXXX " )
                 << "  " << (aK<aI4.tx() ? ToString(aI4.data()[aK]) : " XXXXXX " )
                 << "\n";
   }
}
*/

template <class Type> 
void cTplImInMem<Type>::ReduceGaussienne()
{
    if (mKInOct==0)
    {
         // cTplOctDig<Type>* anOcUp = mTOct.OctUp();
         cOctaveDigeo* anOcUp = mTOct.OctUp();

         if (anOcUp)
         {
              // cTplImInMem<Type> *  aMere = anOcUp->TypedGetImOfSigma(2.0);
              cImInMem *  aMere = anOcUp->GetImOfSigma(2.0);

              VMakeReduce_010(*aMere);
              //MakeReduce( *aMere, mAppli.Params().ReducDemiImage().Val() );
         }
         else
         {
             ELISE_ASSERT(false,"::ReduceGaussienne No OctUp");
         }
         return;
    }

    //==============================================

	double sigma;
	tIm *src;
	if ( !mAppli.doIncrementalConvolution() )
	{
		sigma = sqrt(ElSquare(mResolOctaveBase)-ElSquare(mTMere->mResolOctaveBase));
		src = &mTMere->mIm;
	}
	else
	{
		sigma = mResolOctaveBase;
		src = &mTOct.TypedFirstImage()->mIm;
	}

	mAppli.convolve( *src, sigma, mIm );

/*
    Im1D_REAL8        aRealKerD =  ToRealKernel(aIKerD);
    mKernelTot = Convol(aRealKerD,mTMere->mKernelTot);
*/    


    
}

/****************************************/
/*                                      */
/*             cImInMem                 */
/*                                      */
/****************************************/

/*
void  cImInMem::MakeReduce()
{
    VMakeReduce(*mMere);
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
