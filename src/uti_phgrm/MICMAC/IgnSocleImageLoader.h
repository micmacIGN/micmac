#ifndef __IgnSocleImageLoader_H__
#define __IgnSocleImageLoader_H__

#ifdef __USE_IMAGEIGN__

#include <complex>
#include <vector>
#include "cInterfModuleImageLoader.h"


/**
 Classe d'interface de lecture d'image en JP2000 via la lib Kakadu
 Pour le moment uniquement des images 8 ou 16b
 */
 
class IgnSocleImageLoader: public cInterfModuleImageLoader
	{
	private:
		std::string m_Nomfic;
		
		std::complex<int>		m_SzIm;
		bool                    m_S;
 		eIFImL_TypeNumerique    m_Type;               
		int                     m_Nbc;
		
		
	public:
		typedef std::complex<int> tPInt;
		~IgnSocleImageLoader();		
		IgnSocleImageLoader(std::string const &nomfic);
		
		virtual eIFImL_TypeNumerique PreferedTypeOfResol(int aDeZoom)const;
		virtual std::complex<int> Sz(int aDeZoom)const;
		virtual int NbCanaux()const;		
		virtual void PreparePyram(int aDeZoom);



		void LoadCanalCorrel
                (
                       const sLowLevelIm<float> & anIm,
                       int              aDeZoom,
                       tPInt            aP0Im,
                       tPInt            aP0File,
                       tPInt            aSz
				 );

		
        void LoadNCanaux(const std::vector<sLowLevelIm<float> > & aVImages,
						 int              mFlagLoadedIms,
						 int              aDeZoom,
						 tPInt            aP0Im,
						 tPInt            aP0File,
						 tPInt            aSz
						 );
		
		/** ToDO */
		void LoadNCanaux(const std::vector<sLowLevelIm<short int> > & aVImages,
						 int              mFlagLoadedIms,
						 int              aDeZoom,
						 tPInt            aP0Im,
						 tPInt            aP0File,
						 tPInt            aSz
						 );
		
        void LoadNCanaux(const std::vector<sLowLevelIm<short unsigned int> > & aVImages,
						 int              mFlagLoadedIms,
						 int              aDeZoom,
						 tPInt            aP0Im,
						 tPInt            aP0File,
						 tPInt            aSz
						 );
		/** ToDO */
        void LoadNCanaux(const std::vector<sLowLevelIm<bool> > & aVImages,
						 int              mFlagLoadedIms,
						 int              aDeZoom,
						 tPInt            aP0Im,
						 tPInt            aP0File,
						 tPInt            aSz
						 );
		
		void LoadNCanaux(const std::vector<sLowLevelIm<unsigned char> > & aVImages,
						 int              mFlagLoadedIms,
						 int              aDeZoom,
						 tPInt            aP0Im,
						 tPInt            aP0File,
						 tPInt            aSz
						 );
		
	};
#endif
#endif
