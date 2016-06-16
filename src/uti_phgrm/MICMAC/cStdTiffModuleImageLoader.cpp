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
#ifdef __USE_JP2__
#include "Jp2ImageLoader.h"
#endif
#if __USE_IMAGEIGN__
#include <boost/algorithm/string/predicate.hpp>
#include "IgnSocleImageLoader.h"
#endif
#include "../src/uti_phgrm/MICMAC/MICMAC.h"

GenIm::type_el TypeIMIL2El(eIFImL_TypeNumerique aType)
{
    switch (aType)
    {
          case eUnsignedChar:
               return GenIm::u_int1;

          case eSignedShort:
               return GenIm::int2;

          case eUnsignedShort:
               return GenIm::u_int2;

          case eFloat:
               return GenIm::real4;

          default:;
    }

    ELISE_ASSERT(false,"TypeIMIL2El");
    return GenIm::u_int1;
}

/*************************************************/
/*                                               */
/*              cStdTiffModuleImageLoader        */
/*                                               */
/*************************************************/

template <class aType,class aTypeBase> 
void TplLoadCanalCorrel
     (
           sLowLevelIm<aType>                         anIm,
           Tiff_Im                                    aTif,
           cInterfModuleImageLoader::tPInt            aP0Im,
           cInterfModuleImageLoader::tPInt            aP0File,
           cInterfModuleImageLoader::tPInt            aSz
     )
{
    Fonc_Num aSom = 0;
    Symb_FNum aFonc = aTif.in_proj();
    int aNbc= aTif.nb_chan();
// std::cout << "NB CHAN = " << aNbc << "\n";
    double aSPds =0;
    for (int aK=0 ; aK<aNbc ; aK++)
    {
        double aPds = 1.0;
        aSom = aSom+aFonc.kth_proj(aK)*aPds;
        aSPds += aPds;
    }
   Pt2di aSzIm = Std2Elise(anIm.mSzIm);
   Im2D<aType,aTypeBase> aIm(anIm.mDataLin,anIm.mData,aSzIm.x,aSzIm.y);
   ELISE_COPY
   (
        rectangle(Std2Elise(aP0Im),Std2Elise(aP0Im+aSz)),
        trans(aSom/aSPds,Std2Elise(aP0File-aP0Im)),
        aIm.out()
   );
}

#define StdDefinitMembreLoadCorrel(aType,aTypeBase)\
      void LoadCanalCorrel\
                   (\
                       const sLowLevelIm<aType> & anIm,\
                       int              aDeZoom,\
                       tPInt            aP0Im,\
                       tPInt            aP0File,\
                       tPInt            aSz\
                   )\
{\
   TplLoadCanalCorrel<aType,aTypeBase>(anIm,FileOfResol(aDeZoom),aP0Im,aP0File,aSz);\
}




template <class aType,class aTypeBase> 
void TplLoadNCanaux
     (
           std::vector<sLowLevelIm<aType> > aVIm,
           int              aFlagLoadedIms,
           Tiff_Im          aTif,
           cInterfModuleImageLoader::tPInt            aP0Im,
           cInterfModuleImageLoader::tPInt            aP0File,
           cInterfModuleImageLoader::tPInt            aSz
     )
{
   int aKI=0;
   Output aOutIms = Output::onul(1);
   for (int aBit=0 ; aBit<32 ; aBit++)
   {
       if (aKI<int(aVIm.size()))
       {
          Output aOutLoc =  Output::onul(1);
          if((1<<aBit) & aFlagLoadedIms)
          {
              Pt2di aSzIm = Std2Elise(aVIm[aKI].mSzIm);
              Im2D<aType,aTypeBase> aIm(aVIm[aKI].mDataLin,aVIm[aKI].mData,aSzIm.x,aSzIm.y);
              aOutLoc = aIm.out();
          }
          aOutIms = (aKI==0) ? aOutLoc : Virgule(aOutIms,aOutLoc);
          aKI++;
       }
   }
   ELISE_COPY
   (
        rectangle(Std2Elise(aP0Im),Std2Elise(aP0Im+aSz)),
        trans(aTif.in_proj(),Std2Elise(aP0File-aP0Im)),
        aOutIms
   );
}

#define DefinitStdMembreLoadN(aType,aTypeBase)\
void  LoadNCanaux\
      (\
            const std::vector<sLowLevelIm<aType> > & aVIm,\
            int              aFlag,\
            int              aDeZoom,\
            tPInt            aP0Im,\
            tPInt            aP0File,\
            tPInt            aSz\
       )\
{\
     TplLoadNCanaux<aType,aTypeBase>(aVIm,aFlag,FileOfResol(aDeZoom),aP0Im,aP0File,aSz);\
}


class cStdTiffModuleImageLoader : public cInterfModuleImageLoader
{
   public:

      
      DefinitStdMembreLoadN(unsigned char,int);
      DefinitStdMembreLoadN(short,int);
      DefinitStdMembreLoadN(unsigned short,int);
      DefinitStdMembreLoadN(float,double);

      StdDefinitMembreLoadCorrel(unsigned char,int);
      StdDefinitMembreLoadCorrel(short,int);
      StdDefinitMembreLoadCorrel(unsigned short,int);
      StdDefinitMembreLoadCorrel(float,double);

      // 1
      eIFImL_TypeNumerique PreferedTypeOfResol(int aDeZoom )  const
      {
          switch(FileOfResol(aDeZoom).type_el())
          {
             case GenIm::u_int1 :
                  return eUnsignedChar;
             break; 
             case GenIm::u_int2 :
                  return eUnsignedShort;
             break; 
             case GenIm::real4 :
                  return eFloat;
             break; 
             default:
             break; 
          }
          ELISE_ASSERT(false,"Type Non Traductible::PreferedTypeOfResol");
          return eFloat;
      }
      tPInt Sz(int aDeZoom)  const
      {
            return Elise2Std(FileOfResol(aDeZoom).sz());
      }
      int  NbCanaux()  const
       {
            return FileOfResol(1).nb_chan();
       }

      std::string NameFileOfResol(int aDeZoom) const;
      Tiff_Im     FileOfResol(int aDeZoom) const;
      std::string        CreateFileOfResol(int aDeZoom,bool ForPrepare) const;

      cStdTiffModuleImageLoader
      (
           const cAppliMICMAC & anAppli,
           std::string          aName
      ) :
        mAppli  (anAppli),
        mName   (aName)
      {
      }


      Tiff_Im  StdFile(const std::string & aName) const
      {
          return Tiff_Im::StdConvGen(aName,1,mAppli.Correl16Bits().ValWithDef(true));
      }

      void PreparePyram(int aDeZoom) 
      {
// std::cout << "PreparePyramPreparePyram " << aDeZoom << "\n";
         CreateFileOfResol(aDeZoom,true);
      }
      std::string  NameTiffImage() const;
   private:

      void  CalcFilter();

      const cAppliMICMAC & mAppli;
      std::string    mName;

};



std::string  cStdTiffModuleImageLoader::NameTiffImage() const
{
   return NameFileOfResol(1);
}


std::string cStdTiffModuleImageLoader::NameFileOfResol(int aDeZoom) const
{
   if (aDeZoom == 1)
   {
      return    mAppli.DirImagesInit()
             +  mName;
   }

   return    mAppli.FullDirPyr()
          +  mAppli.NameFilePyr(mName,aDeZoom);
}

//  [1]- Si le fichier existe on le renvoie
//  [2]- Sinon si la resolution vaut 1 y un pb
//  [3]- Si la resolution n'est pas une puissance de 2 y a  aussi un pb
//  [4]- Sinon on le calcul par reduction de la resolution
//     du dessus.

Tiff_Im     cStdTiffModuleImageLoader::FileOfResol(int aDeZoom) const
{
   std::string aName =   CreateFileOfResol(aDeZoom,false);
   mAppli.VerifSzFile(StdFile(aName).sz());
   return StdFile(aName);
}

std::string     cStdTiffModuleImageLoader::CreateFileOfResol(int aDeZoom,bool ForPrepare) const
{
    cEl_GPAO * aGPAO = mAppli.GPRed2();


    // [1]
    std::string aName = NameFileOfResol(aDeZoom);

    if (ELISE_fp::exist_file(aName))
    {
       if (aGPAO)
          aGPAO->GetOrCreate(aName,"");
       return aName;
    }

    // [2]
    if (aDeZoom==1)
    {
        std::cout << "FILE= " << aName << "\n";
        ELISE_ASSERT(false,"Pas de fichier image original en resolution 1");
    }

    // [3]
    ELISE_ASSERT
    (
         is_pow_of_2(aDeZoom),
         "La resolution image n'est pas une puissance de 2"
   );


   /* [4]

       Rappel pour diviser par 2 la resolution (fonction MakeTiffRed2),
       on prend un pixel sur 4, et on lui donne la valeur de l'image
       initiale par le filtre  :
                    1  2  1
                    2  4  2
                    1  2  1
       (c'est un modele separable et centre, qui fait une image
       pas trop piquee).

       Ensuite les valeur sont divisee par 16 en general, mais
       on peut pour avoir un diviseur plus petit pour obtenir
       une plus grande precision tant que l'on est sur de ne
       pas saturer. C'est le role du parametre DivIm de
       de TypePyramImage.


       Le principe de TypePyramImage est que :

          - chaque specification de type, a une resolution donnee,
          est etendue  pour les resolutions inferieure

         - l'eventuel coefficient multiplicatif ne vaut que pour
        la resolution specifiee

#
#
        Par exemple , avec une image a resolution 1 sur un octet et
        les valeurs suivantes dans TypePyramImage :

          Resol 4 , Div 1, type eUInt16Bits
          Resol 8 , Div 1, type eUInt16Bits
          Resol 32 , type    eFloat32Bits

        On aura les types suivants dans la pyramide:

         Resol 1     : eUInt8Bits ;
         Resol 2     : eUInt8Bits ;

         Resol 4     : eUInt16Bits ;  les valeur seront ne seront
                      pas divisee par 16 apres convolution pour passer de R2 a R4;
         Resol 8     : eUInt16Bits ;  valeur non divisees par 16 ;
         Resol 16 :    eUInt16Bits

         Resol 32  : eFloat32Bits
         Resol 64,128 ...  : eFloat32Bits


   */


   // On recherche la plus faible resolution meilleure
   // ou egale a aDeZoom, c'est elle qui fixe le type,
   // si elle vaut exactement aDeZoom elle fixe aussi le Diviseur

        // Auparavant , initialisation sur les valeur de la resolution 1
   Tiff_Im aFile1 = FileOfResol(1);
   GenIm::type_el aType = aFile1.type_el();
   int aDivIm = 16;

   if (mAppli.HighPrecPyrIm().Val())
   {
        if (aType==GenIm::u_int1)
        {
            if (aDeZoom==2) 
            {
                aType = GenIm::u_int2;
                aDivIm = 1;
            }
            else if (aDeZoom==4) 
            {
                aType = GenIm::u_int2;
                aDivIm = 2;
            }
            else 
            {
                aType = GenIm::real4;
            }
        }
        else 
        {
                aType = GenIm::real4;
                aDivIm = 1;
        }
   }

   int aResolSup =1;

   for
   (
       cAppliMICMAC::tCsteIterTPyr iTP = mAppli.TypePyramImage().begin();
       iTP != mAppli.TypePyramImage().end();
       iTP++
   )
   {
        if ((iTP->Resol() <= aDeZoom)  && (iTP->Resol()>aResolSup))
        {
             aResolSup = iTP->Resol();
             if (iTP->Resol() == aDeZoom)
                aDivIm = iTP->DivIm().Val();
             switch (iTP->TypeEl())
             {
                  case eUInt8Bits :
                          aType = GenIm::u_int1;
                  break;

                  case eUInt16Bits :
                          aType = GenIm::u_int2;
                  break;

                  case eFloat32Bits :
                          aType = GenIm::real4;
                  break;
                  default :
                      ELISE_ASSERT(false,"Incoherence dans cPriseDeVue::FileOfResol");
                  break;
             }
        }
   }

   // Force la creation de la resolution du dessus
   // Genere la reduction d'un facteur 2
   std::string aNameIn =  (aDeZoom==2) ?
                          NameFileStd(NameFileOfResol(aDeZoom/2),1,mAppli.Correl16Bits().ValWithDef(true)) :
                          NameFileOfResol(aDeZoom/2)
                         ;
   std::cout << "<< Make Pyram for " << aNameIn  << " " << ForPrepare << " " << aGPAO<< "\n";
   if (aGPAO && ForPrepare)
   {
       CreateFileOfResol(aDeZoom/2,ForPrepare);
       std::string aCom =    MM3dBinFile_quotes( "Reduc2MM" )
                           + protect_spaces(aNameIn) + " "
                           + protect_spaces(aName) + " " 
                           + ToString(int(aType)) + " "
                           + ToString(int(aDivIm)) + " "
                           + ToString(mAppli.HasVSNI()) + " "
                           + ToString(mAppli.VSNI())    + " ";
       aGPAO->GetOrCreate(aName,aCom);

       aGPAO->TaskOfName("all").AddDep(aName);
       if (aDeZoom!=2)
       {
          aGPAO->TaskOfName(aName).AddDep(aNameIn);
       }
   }
   else
   {
       FileOfResol(aDeZoom/2);
       MakeTiffRed2 ( aNameIn,  aName, aType, aDivIm, mAppli.HasVSNI(), mAppli.VSNI());
       std::cout << ">> Done Pyram for " << NameFileOfResol(aDeZoom/2) << "\n";
   }

   return aName;

}

/*************************************************/
/*                                               */
/*              cInterfModuleImageLoader         */
/*                                               */
/*************************************************/

// A mettre dans le .h pour utilisation hors MICMAC
/*
    cInterfModuleImageLoader::cInterfModuleImageLoader() :
    mAppli (0)
{
}
*/
cAppliMICMAC & cInterfModuleImageLoader::Appli()
{
   ELISE_ASSERT(mAppli!=0,"cInterfModuleImageLoader, Appli Not Init");
   return *mAppli;
}

void  cInterfModuleImageLoader::SetAppli(cAppliMICMAC * anAppli)
{
    mAppli = anAppli;
}

std::string  cInterfModuleImageLoader::NameTiffImage() const
{
    ELISE_ASSERT(false,"Pas de cInterfModuleImageLoader::NameTiffImage");
    return "";
}


std::string cInterfModuleImageLoader::NameFileOfResol(int aDeZoom) const
{
    ELISE_ASSERT(false,"Pas de cInterfModuleImageLoader::NameFileOfResol");
    return "";
}



/*************************************************/
/*                                               */
/*              cAppliMICMAC                     */
/*                                               */
/*************************************************/

cInterfModuleImageLoader *  cAppliMICMAC::LoaderFiltre
      (
         cInterfModuleImageLoader * aLoaderBase,
         std::string                aName
      )
{
   bool useFilter = false;
   for 
   (
      std::list<cSpecFitrageImage>::const_iterator itF = FiltreImageIn().begin();
      itF != FiltreImageIn().end();
      itF++
   )
   {

       if (
                 (! itF->PatternSelFiltre().IsInit())
              || (itF->PatternSelFiltre().Val()->Match(aName))
          )
       {
          useFilter = true;
       }
   }

   if (!useFilter)
   {
      return aLoaderBase;
   }

   std::string aNameZ1 =   StdPrefix(aName) 
                          + "_MicMac_Filtered.tif";
   std::string aFNameZ1 =   DirImagesInit()  + aNameZ1;
  
   if ( ELISE_fp::exist_file(aFNameZ1))
      return new cStdTiffModuleImageLoader(*this,aNameZ1);


   Pt2di aSz = Std2Elise(aLoaderBase->Sz(1));
   Tiff_Im * aTifZ1 = 0;
   int aSzBlocY = round_ni(1e8/aSz.x);
   int aBrd = ElMin(200,round_ni(aSzBlocY*0.05));

   cDecoupageInterv1D aDec1
                      (
                          cInterv1D<int>(0,aSz.y),
                          aSzBlocY,
                          cInterv1D<int>(-aBrd,aBrd)
                      );


   for (int aKInt = 0 ; aKInt<aDec1.NbInterv() ; aKInt++)
   {
        cInterv1D<int> aIntIn  = aDec1.KthIntervIn(aKInt);
        cInterv1D<int> aIntOut = aDec1.KthIntervOut(aKInt);

        Im2D_REAL4 aIm(aSz.x,aIntIn.Larg());
        LoadAllImCorrel
        (
            aIm,
            aLoaderBase,
            1,
            Pt2di(0,aIntIn.V0())
        );


       Fonc_Num aFonc = aIm.in_proj();


       for 
       (
          std::list<cSpecFitrageImage>::const_iterator itF = FiltreImageIn().begin();
          itF != FiltreImageIn().end();
          itF++
       )
       {
          if (
                 (! itF->PatternSelFiltre().IsInit())
              || (itF->PatternSelFiltre().Val()->Match(aName))
             )
           { 
                aFonc = FiltrageImMicMac(*itF,aFonc,1,1.0);
/*
                aFonc = deriche(aFonc.v0(),1/itF->SzFiltre());
                aFonc = polar(aFonc,0).v0();
*/
           }
       }
       if (aTifZ1==0)
       {
           GenIm::type_el aType = TypeIMIL2El(aLoaderBase->PreferedTypeOfResol(1));
	   if (!aFonc.integral_fonc(true))
	      aType = GenIm::real4;
           aTifZ1 = new Tiff_Im
                        (
                            aFNameZ1.c_str(),
                            aSz,
                            aType,
                            Tiff_Im::No_Compr,
                            Tiff_Im::BlackIsZero
                     );
       }

       ELISE_COPY
       (
           rectangle
           (
             Pt2di(0, aIntOut.V0()),
             Pt2di(aSz.x, aIntOut.V1())
           ),
           trans(aFonc,Pt2di(0,-aIntIn.V0())),
           aTifZ1->out()
       );
   }
   delete aTifZ1;
   return new cStdTiffModuleImageLoader(*this,aNameZ1);
}


cInterfModuleImageLoader * cAppliMICMAC::GetMIL
                           (
                               cGeometrieImageComp * aGIC,
                               const std::string & aName
                           )
{


 const cTplValGesInit< cModuleImageLoader > &  aMIL = aGIC->ModuleImageLoader();
	cInterfModuleImageLoader * aRes = 0;
 if ( ! aMIL.IsInit())
 {
	 //on recupere l'extension
	 int placePoint = -1;
	 for(int l=(int)(aName.size() - 1);(l>=0)&&(placePoint==-1);--l)
	 {
		 if (aName[l]=='.')
		 {
			 placePoint = l;
		 }
	 }
	 std::string ext = std::string("");
	 if (placePoint!=-1)
	 {
		 ext.assign(aName.begin()+placePoint+1,aName.end());
	 }
	 //std::cout << "Extension : "<<ext<<std::endl;
	 
#if defined (__USE_JP2__)
	// on teste l'extension
	if ((ext==std::string("jp2")) || 
		(ext==std::string("JP2")) || 
		(ext==std::string("Jp2")))
	{
		aRes = new JP2ImageLoader(DirImagesInit()+aName);
	}
#endif
#if defined (__USE_IMAGEIGN__)
	 // on teste l'extension
	 if ((aRes==NULL) && (boost::algorithm::iequals(ext,std::string("jp2"))|| 
						 boost::algorithm::iequals(ext,std::string("ecw")) || 
						 boost::algorithm::iequals(ext,std::string("jpg")) || 
						 boost::algorithm::iequals(ext,std::string("dmr")) || 
						 boost::algorithm::iequals(ext,std::string("bil"))))
	 {
		 aRes = new IgnSocleImageLoader(DirImagesInit()+aName);
	 }
#endif
    if (aRes==NULL)
		aRes = new cStdTiffModuleImageLoader(*this,aName);
 }
 else
 {
     cLibDynAllocator<cInterfModuleImageLoader> 
          mAlloc
          (
              *this,
               aMIL.Val().NomModule(),
               "create_micmac_image_loader"
          );

     aRes =  mAlloc.AllocObj
          (
               DirImagesInit() +  aName,
               aMIL.Val().NomLoader()
          );
  }
  aRes->SetAppli(this);

  cInterfModuleImageLoader * aResFiltre = LoaderFiltre(aRes,aName);
  if (aResFiltre!=aRes)
  {
       aResFiltre->SetAppli(this);
       delete aRes;
       aRes = aResFiltre;
  }
  return aRes;
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
