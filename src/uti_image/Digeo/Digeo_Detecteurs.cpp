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

/****************************************/
/*                                      */
/*           cTplImInMem                */
/*                                      */
/****************************************/

template <class Type> typename cTplImInMem<Type>::tBase cTplImInMem<Type>::DOG(Type *** aC,const Pt3di& aP)
{
     return aC[aP.z][aP.y][aP.x]-aC[aP.z+1][aP.y][aP.x];
}

template <class Type> bool cTplImInMem<Type>::SupDOG(Type *** aC,const Pt3di& aP1,const Pt3di& aP2)
{
    tBase aV1 = DOG(aC,aP1);
    tBase aV2 = DOG(aC,aP2);

    if (aV1 > aV2) return true;
    if (aV1 < aV2) return false;

    if (aP1.z > aP2.z) return true;
    if (aP1.z < aP2.z) return false;

    if (aP1.y > aP2.y) return true;
    if (aP1.y < aP2.y) return false;

    if (aP1.x > aP2.x) return true;
    if (aP1.x < aP2.x) return false;

    ELISE_ASSERT(false,"::SupDOG");
    return true;
}

template <class Type> void cTplImInMem<Type>::ExtramDOG(Type *** aC,const Pt2di & aP,bool & isMax,bool & isMin)
{
    isMax = true;
    isMin = true;

    for (int aZ=-1 ; aZ<=1 ; aZ++)
    {
       for (int aY=-1 ; aY<=1 ; aY++)
       {
          for (int aX=-1 ; aX<=1 ; aX++)
          {
              if (aZ||aY||aX)
              {
                    if (SupDOG(aC,Pt3di(aP.x,aP.y,0),Pt3di(aP.x+aX,aP.y+aY,aZ)))
                       isMin = false;
                    else
                       isMax = false;
                    if ((!isMin) && (!isMax))
                       return;
              }
          }
       }
    }
}

/*
   Autre facon de voir les choses :


    I(x,y,z) = I0 +  aX + bY + cZ + d X2 + ....  Z2

    On resout au moinde carre

*/
template <class Type> void LsqAffineDiff
                      (
                           Type *** aC,
                           int      anX,
                           int      anY,
                           int      aNiv
     )
{
    for (int aDs = -1 ; aDs <= -1 ; aDs++)
    {
    }
}

template <class Type>
eTypeExtreSift cTplImInMem<Type>::CalculateDiff_none(
	tBase *prevDoG, // previous,
	tBase *currDoG, // current
	tBase *nextDoG, // and next DoG at coordinates anX, anY
	int      anX,
	int      anY,
	int      aNiv )
{
	mTrX = mTrY = mTrS = 0;
	return eTES_Ok;
}

template <class Type>
eTypeExtreSift cTplImInMem<Type>::CalculateDiff_2d
     (
          tBase *prevDoG, // previous,
          tBase *currDoG, // current
          tBase *nextDoG, // and next DoG at coordinates anX, anY
          int      anX,
          int      anY,
          int      aNiv
     )
{
    mIx = anX;
    mIy = anY;

    mGX =  (currDoG[mN4]-currDoG[mN3])/2.0;
    mGY =  (currDoG[mN6]-currDoG[mN1])/2.0;

    mDxx = currDoG[mN4]+currDoG[mN3] - 2*currDoG[0];
    mDyy = currDoG[mN6]+currDoG[mN1] - 2*currDoG[0];

    mDxy = (currDoG[mN7]+currDoG[mN0]-currDoG[mN5]-currDoG[mN2])/4.0;

    double aDelta = mDxx * mDyy - ElSquare(mDxy);

    if ( aDelta<=numeric_limits<double>::epsilon() ) return eTES_instable_unsolvable;

    mTrX = ( mDxy*mGY-mDyy*mGX )/aDelta;
    mTrY = ( mDxy*mGX-mDxx*mGY )/aDelta;
    mTrS = 0;

    #ifdef __DEBUG_DIGEO
       double e = numeric_limits<float>::epsilon();
       double verif1 = mDxx*mTrX + mDxy*mTrY,
       verif2 = mDxy*mTrX + mDyy*mTrY,
       score1 = verif1+mGX,
       score2 = verif2+mGY;
       if ( score1>e || score1<-e ){
           cerr << "DEBUG: verif1 = " << verif1 << " -dx = " << -mGX << " score = " << score1 << endl;
           exit( EXIT_FAILURE );
       }
       if ( score2>e || score2<-e ){
           cerr << "DEBUG: verif2 = " << verif2 << " -dy = " << -mGY << " score = " << score2 << endl;
           exit( EXIT_FAILURE );
       }
    #endif

    int aDx = round_ni(mTrX);
    int aDy = round_ni(mTrY);

    if ((aDx!=0) || (aDy!=0))
    {
         if (aNiv>=3)
         {
            return eTES_instable_tooDeepRecurrency;
         }

         anX += aDx;
         anY += aDy;

         if (     (anX>=mBrd)
              &&  (anX<mSz.x-mBrd)
              &&  (anY>=mBrd)
              &&  (anY<mSz.y-mBrd)
            )
         {
             int offset = aDx+aDy*mSz.x;
             return CalculateDiff_2d( prevDoG+offset, currDoG+offset, nextDoG+offset, anX, anY, aNiv+1 );
         }
         else
            return eTES_instable_outOfImageBound;
    }


    double aTrace = mDxx + mDyy;
    double aRatio = ElSquare(aTrace) / aDelta;

    //if (ElSquare(mGX)+ElSquare(mGY)<mSeuilGrad)
    double val = currDoG[0]+0.5*( mTrX*mGX+mTrY*mGY );
    if ( val<0 ) val=-val;
    if ( val<=(0.02/3)*mImGlob.GetMaxValue() )
       return eTES_GradFaible;

    //if (aRatio > mSeuilTr2Det)
    if ( aRatio>=12.1*mImGlob.GetMaxValue()*mImGlob.GetMaxValue() )
       return eTES_TropAllonge;

   return eTES_Ok;
}

#include "../Sift/Gauss34.h"

template <class Type>
eTypeExtreSift cTplImInMem<Type>::CalculateDiff_3d
     (
          tBase *prevDoG, // previous,
          tBase *currDoG, // current
          tBase *nextDoG, // and next DoG at coordinates anX, anY
          int      anX,
          int      anY,
          int      aNiv
     )
{
    mIx = anX;
    mIy = anY;

    mGX =  ( currDoG[mN4]-currDoG[mN3] )/2.;
    mGY =  ( currDoG[mN6]-currDoG[mN1] )/2.;
    mGS =  ( nextDoG[0]-prevDoG[0] )/2.;
    mDxx = currDoG[mN4]+currDoG[mN3]-2.*currDoG[0];
    mDyy = currDoG[mN6]+currDoG[mN1]-2.*currDoG[0];
    mDss = nextDoG[0]+prevDoG[0]-2.*currDoG[0];
    mDxy = ( currDoG[mN7]+currDoG[mN0]-currDoG[mN5]-currDoG[mN2] )/4.;
    mDxs = ( nextDoG[mN4]+prevDoG[mN3]-nextDoG[mN3]-prevDoG[mN4] )/4.;
    mDys = ( nextDoG[mN6]+prevDoG[mN1]-nextDoG[mN1]-prevDoG[mN6] )/4.;

    double m[12];
    double b[3];
    m[3]  = b[0] = -mGX;
    m[7]  = b[1] = -mGY;
    m[11] = b[2] = -mGS;
    m[0]  = mDxx;
    m[5]  = mDyy;
    m[10] = mDss;
    m[1] = m[4] = mDxy;
    m[2] = m[8] = mDxs;
    m[6] = m[9] = mDys;

    int aDx, aDy;
    //double aDelta = mDxx*( mDyy*mDss-ElSquare(mDys) ) - mDxy*( mDxy*mDss-mDys*mDxs ) + mDxs*( mDxy*mDys-mDxs*mDyy );
    //if ( aDelta>numeric_limits<double>::epsilon() ){
		 siftpp__gauss33_invert_b( m, b );
		 mTrX = b[0];
		 mTrY = b[1];
		 mTrS = b[2];

		 #ifdef __DEBUG_DIGEO
			 double failed = false;
			 double e = numeric_limits<float>::epsilon();
			 double verif1 = mDxx*mTrX + mDxy*mTrY + mDxs*mTrS,
					  verif2 = mDxy*mTrX + mDyy*mTrY + mDys*mTrS,
					  verif3 = mDxs*mTrX + mDys*mTrY + mDss*mTrS,
					  score1 = verif1+mGX,
					  score2 = verif2+mGY,
					  score3 = verif3+mGS;
			 double aDelta = mDxx*( mDyy*mDss-ElSquare(mDys) ) - mDxy*( mDxy*mDss-mDys*mDxs ) + mDxs*( mDxy*mDys-mDxs*mDyy );
			 if ( aDelta>e ){
				 if ( score1>e || score1<-e ){ cerr << "DEBUG: verif1 = " << verif1 << " -dx = " << -mGX << " score = " << score1 << endl; failed=true; }
				 if ( score2>e || score2<-e ){ cerr << "DEBUG: verif2 = " << verif2 << " -dy = " << -mGY << " score = " << score2 << endl; failed=true; }
				 if ( score3>e || score3<-e ){ cerr << "DEBUG: verif3 = " << verif3 << " -ds = " << -mGS << " score = " << score3 << endl; failed=true; }
				 if ( failed ) exit( EXIT_FAILURE );
			 }
		 #endif

		 //int aDx = round_ni(mTrX);
		 //int aDy = round_ni(mTrY);
		 aDx = ( mTrX>0.6?1:0 );
		 aDy = ( mTrY>0.6?1:0 );
		 if ( mTrX<-0.6 ) aDx=-1;
		 if ( mTrY<-0.6 ) aDy=-1;

		if ( ( (aDx!=0) || (aDy!=0) ) && aNiv<4 ){
			anX += aDx;
			anY += aDy;
			if ( ( anX>=mBrd ) && ( anX<mSz.x-mBrd ) &&
				  ( anY>=mBrd ) && ( anY<mSz.y-mBrd ) ){
				int offset = aDx+aDy*mSz.x;

				unsigned char *used = mUsed_points_map+( anX+anY*mSz.x );
				if ( *used!=0 ) return eTES_AlreadyComputed;
				*used = 1;

				return CalculateDiff_3d( prevDoG+offset, currDoG+offset, nextDoG+offset, anX, anY, aNiv+1 );
			}
			else
				return eTES_instable_outOfImageBound;
		}
	//}
	
    const double deltaThreshold = 1.5;
    if ( abs(mTrX)>=deltaThreshold || abs(mTrY)>=deltaThreshold || abs(mTrS)>=deltaThreshold ) return eTES_displacementTooBig;
    
    double aTrace = mDxx + mDyy;
    double aRatio = ElSquare(aTrace)/( mDxx*mDyy-ElSquare(mDxy) );

    //if (ElSquare(mGX)+ElSquare(mGY)<mSeuilGrad)
    double val = currDoG[0]+0.5*( mTrX*mGX+mTrY*mGY+mTrS*mGS );

    double s = (double)mKInOct+mTrS;
    if ( s<0 || s>=(double)mOct.lastLevelIndex() ) return eTES_instable_outOfScaleBound;
    
    if ( val<0 ) val=-val;
    if ( val<=(0.02/3)*mImGlob.GetMaxValue() ) return eTES_GradFaible;

    //if (aRatio > mSeuilTr2Det)
    if ( aRatio<0 || aRatio>=12.1*mImGlob.GetMaxValue()*mImGlob.GetMaxValue() ) return eTES_TropAllonge;

   return eTES_Ok;
}

template <class Type> 
void cTplImInMem<Type>::ExtractExtremaDOG
     (
          const cSiftCarac & aSC,
          cTplImInMem<Type> & aPrec,
          cTplImInMem<Type> & aNext
     )
{
	// allocate the map of already used point
	const int nbPix = mSz.x*mSz.y;
	mUsed_points_map = new unsigned char[nbPix];
	// the map is empty for now
	memset( mUsed_points_map, 0, nbPix );

    //Type strengthThreshold = Type( (0.02/mOct.NbImOri())*mImGlob.GetMaxValue() );
    const Type strengthThreshold = 2./375.;

   double aRalm = aSC.RatioAllongMin().Val();
   mSeuilTr2Det = (aRalm+1)*(1+1/aRalm);
   mSeuilGrad = ElSquare(mImGlob.GradMoyCorrecDyn()*(aSC.RatioGrad().Val()/mResolOctaveBase));

   ELISE_ASSERT( (mSz==aPrec.mSz)&&(mSz==aNext.mSz), "Size im diff in ::ExtractExtremaDOG" );

    mNbExtre = 0;
    mNbExtreOK = 0;

    bool doMax = aSC.DoMax().Val();
    bool doMin = aSC.DoMin().Val();

    mBrd = 1;
    int brd_2 = 2*mBrd;
    tBase *itDoG = DoG()+mBrd*(mN7),
          *itPrevDoG = aPrec.DoG()+mBrd*(mN7),
          *itNextDoG = aNext.DoG()+mBrd*(mN7);
    int aX1 = mSz.x-mBrd;
    for (int anY=mBrd ; anY<(mSz.y-mBrd) ; anY++)
    {
        for (int anX=mBrd; anX<aX1 ; anX++)
        {
				unsigned char *used = mUsed_points_map+( anX+anY*mSz.x );
				bool isAlreadyComputed = true;
				if ( *used==0 ){ *used=1; isAlreadyComputed=false; }
            
            mDogPC = *itDoG;
            bool isMin=false;
            mResDifSift = eTES_Uncalc;

            if ( doMax

                 && mDogPC>=strengthThreshold

                 && ( mDogPC > itPrevDoG[mN0] )
                 && ( mDogPC > itPrevDoG[mN1] )
                 && ( mDogPC > itPrevDoG[mN2] )
                 && ( mDogPC > itPrevDoG[mN3] )
                 && ( mDogPC > itPrevDoG[0] )
                 && ( mDogPC > itPrevDoG[mN4] )
                 && ( mDogPC > itPrevDoG[mN5] )
                 && ( mDogPC > itPrevDoG[mN6] )
                 && ( mDogPC > itPrevDoG[mN7] )

                 && ( mDogPC > itDoG[mN0] )
                 && ( mDogPC > itDoG[mN1] )
                 && ( mDogPC > itDoG[mN2] )
                 && ( mDogPC > itDoG[mN3] )
                 && ( mDogPC > itDoG[mN4] )
                 && ( mDogPC > itDoG[mN5] )
                 && ( mDogPC > itDoG[mN6] )
                 && ( mDogPC > itDoG[mN7] )

                 && ( mDogPC > itNextDoG[mN0] )
                 && ( mDogPC > itNextDoG[mN1] )
                 && ( mDogPC > itNextDoG[mN2] )
                 && ( mDogPC > itNextDoG[mN3] )
                 && ( mDogPC > itNextDoG[0] )
                 && ( mDogPC > itNextDoG[mN4] )
                 && ( mDogPC > itNextDoG[mN5] )
                 && ( mDogPC > itNextDoG[mN6] )
                 && ( mDogPC > itNextDoG[mN7] )
               )
            {
                //mResDifSift = CalculateDiff_2d( itPrevDoG, itDoG, itNextDoG, anX,anY,0 );
                //mResDifSift = CalculateDiff_3d( itPrevDoG, itDoG, itNextDoG, anX, anY, 0 );
                mResDifSift = (this->*mCalculateDiff)( itPrevDoG, itDoG, itNextDoG, anX,anY, 0 );
            }
            else if ( doMin

                     && mDogPC<=-strengthThreshold

                     && ( mDogPC < itPrevDoG[mN0] )
                     && ( mDogPC < itPrevDoG[mN1] )
                     && ( mDogPC < itPrevDoG[mN2] )
                     && ( mDogPC < itPrevDoG[mN3] )
                     && ( mDogPC < itPrevDoG[0] )
                     && ( mDogPC < itPrevDoG[mN4] )
                     && ( mDogPC < itPrevDoG[mN5] )
                     && ( mDogPC < itPrevDoG[mN6] )
                     && ( mDogPC < itPrevDoG[mN7] )

                     && ( mDogPC < itDoG[mN0] )
                     && ( mDogPC < itDoG[mN1] )
                     && ( mDogPC < itDoG[mN2] )
                     && ( mDogPC < itDoG[mN3] )
                     && ( mDogPC < itDoG[mN4] )
                     && ( mDogPC < itDoG[mN5] )
                     && ( mDogPC < itDoG[mN6] )
                     && ( mDogPC < itDoG[mN7] )

                     && ( mDogPC < itNextDoG[mN0] )
                     && ( mDogPC < itNextDoG[mN1] )
                     && ( mDogPC < itNextDoG[mN2] )
                     && ( mDogPC < itNextDoG[mN3] )
                     && ( mDogPC < itNextDoG[0] )
                     && ( mDogPC < itNextDoG[mN4] )
                     && ( mDogPC < itNextDoG[mN5] )
                     && ( mDogPC < itNextDoG[mN6] )
                     && ( mDogPC < itNextDoG[mN7] )
                   )
            {
                isMin=true;
                //mResDifSift = CalculateDiff_2d( itPrevDoG, itDoG, itNextDoG, anX,anY,0 );
                //mResDifSift = CalculateDiff_3d( itPrevDoG, itDoG, itNextDoG, anX,anY, 0 );
                mResDifSift = (this->*mCalculateDiff)( itPrevDoG, itDoG, itNextDoG, anX,anY, 0 );
            }

            if ( isAlreadyComputed ) mResDifSift = eTES_AlreadyComputed;
            
            if (mResDifSift==eTES_Ok)
            {
                Pt2dr aP(mIx+mTrX,mIy+mTrY);
                if ( mOct.Pt2Sauv(aP) ){
                    double localScale = mImGlob.Sigma0()*pow( 2, ( mKInOct+mTrS )/mOct.NbImOri() );
                    mFeaturePoints.push_back( cPtsCaracDigeo( aP, mOct.trueSamplingPace()*localScale, localScale, ( isMin?eSiftMinDog:eSiftMaxDog ) ) );
                }
            }

            itDoG++; itPrevDoG++; itNextDoG++;
        }
        itDoG+=brd_2; itPrevDoG+=brd_2; itNextDoG+=brd_2;
    }

	// free the map of already used point
	delete [] mUsed_points_map;
	mUsed_points_map = NULL;
}

string eToString( const eTypeExtreSift &i_enum )
{
	switch ( i_enum ){
	case eTES_Uncalc: return "eTES_Uncalc";
	case eTES_instable_unsolvable: return "eTES_instable_unsolvable";
	case eTES_instable_tooDeepRecurrency: return "eTES_instable_tooDeepRecurrency";
	case eTES_instable_outOfImageBound: return "eTES_instable_outOfImageBound";
	case eTES_instable_outOfScaleBound: return "eTES_instable_outOfScaleBound";
	case eTES_displacementTooBig: return "eTES_displacementTooBig";
	case eTES_GradFaible: return "eTES_GradFaible";
	case eTES_TropAllonge: return "eTES_TropAllonge";
	case eTES_AlreadyComputed: return "eTES_AlreadyComputed";
	case eTES_Ok: return "eTES_Ok";
	}
	std::cout << "Enum = eTypeExtreSift" << endl;
	ELISE_ASSERT( false, "Bad Value in eToString for enum value " );
   return "";
}

/*
InstantiateFunctionTplDigeo(REAL4,REAL8)
InstantiateFunctionTplDigeo(U_INT2,INT)

*/
template <> INT**    cTplImInMem<U_INT2>::theMDog = 0;
template <> double** cTplImInMem<REAL4>::theMDog = 0;

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
