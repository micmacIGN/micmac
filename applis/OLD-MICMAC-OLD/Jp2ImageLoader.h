#ifndef __JP2ImageLoader_H__
#define __JP2ImageLoader_H__

#ifdef __USE_JP2__

#include <complex>
#include <vector>
#include "cInterfModuleImageLoader.h"


namespace NS_ParamMICMAC
{
/**
 Classe d'interface de lecture d'image en JP2000 via la lib Kakadu
 Pour le moment uniquement des images 8 ou 16b
 */
 
class JP2ImageLoader: public cInterfModuleImageLoader
	{
	private:
		std::string m_Nomfic;
		
		std::complex<int> 	m_SzIm;
		bool                    m_S;
		int                     m_BPS;
 		eTypeNumerique          m_Type;               
		int                     m_Nbc;
		
		
	public:
		typedef std::complex<int> tPInt;
		~JP2ImageLoader()
		{
		}
		
		JP2ImageLoader(std::string const &nomfic);
		
		virtual eTypeNumerique PreferedTypeOfResol(int aDeZoom)const
		{
			return m_Type;
		}
		virtual std::complex<int> Sz(int aDeZoom)const
		{
			return std::complex<int>(m_SzIm.real()/aDeZoom,m_SzIm.imag()/aDeZoom);
		}               
		virtual int NbCanaux()const
		{       
			return m_Nbc;
		}
		
		virtual void PreparePyram(int aDeZoom){}



		void LoadCanalCorrel
                (
                       const sLowLevelIm<float> & anIm,
                       int              aDeZoom,
                       tPInt            aP0Im,
                       tPInt            aP0File,
                       tPInt            aSz
                )
		{
			std::vector<sLowLevelIm<float> > anImNCanaux;
			for(int i=0;i<m_Nbc;++i)
			{
				float * DataLin = new  float [(unsigned long)anIm.mSzIm.real()*(unsigned long)anIm.mSzIm.imag()];
				float ** Data = new  float* [anIm.mSzIm.imag()];
				for(int l=0;l<anIm.mSzIm.imag();++l)
				{
					Data[l] = DataLin + l*anIm.mSzIm.real();
				}
				anImNCanaux.push_back(sLowLevelIm<float>(DataLin,Data,anIm.mSzIm));
			}	
			LoadNCanaux(anImNCanaux,0,aDeZoom,aP0Im,aP0File,aSz);
			for(int l=0;l<aSz.imag();++l)
			{
				float * pt_out = anIm.mData[l+aP0Im.imag()]+aP0Im.real();
				std::vector<float*> vpt_in;
				for(int i=0;i<m_Nbc;++i)
				{
					vpt_in.push_back(anImNCanaux[i].mData[l+aP0Im.imag()]+aP0Im.real());
				}
				for(int c=0;c<aSz.real();++c)
				{
					(*pt_out)=(float)0.;
					for(size_t n=0;n<vpt_in.size();++n)
					{
						(*pt_out)+=(*vpt_in[n]);
						++vpt_in[n];				
					}
					(*pt_out)/=(float)m_Nbc;
					++pt_out;		
				}	
			}
			for(int i=0;i<m_Nbc;++i)
                        {
                                delete[] anImNCanaux[i].mDataLin;
                                delete[] anImNCanaux[i].mData;
			}	
		}

		
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
						 )
        {std::cout << "ERREUR LoadNCanaux en short int non implemente" << std::endl;}
		
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
						 )
        {std::cout << "ERREUR LoadNCanaux en bool non implemente" << std::endl;}
		
		void LoadNCanaux(const std::vector<sLowLevelIm<unsigned char> > & aVImages,
						 int              mFlagLoadedIms,
						 int              aDeZoom,
						 tPInt            aP0Im,
						 tPInt            aP0File,
						 tPInt            aSz
						 );
		
	};
};
#endif
#endif
