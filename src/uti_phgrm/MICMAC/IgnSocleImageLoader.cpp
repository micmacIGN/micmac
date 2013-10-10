#ifdef __USE_IMAGEIGN__

#include "StdAfx.h"

#include <iostream>
#include <vector>
#include <complex>

#include <boost/filesystem.hpp>
#include "IgnSocleImageLoader.h"
#include <ign/image/io/ImageInput.h>
#include <ign/image/BufferImage.h>

	
#define DefineMembreLoadN(aType)\
void IgnSocleImageLoader::LoadNCanaux\
(\
const std::vector<sLowLevelIm<aType> > & aVImages,\
int              mFlagLoadedIms,\
int              aDeZoom,\
tPInt            aP0Im,\
tPInt            aP0File,\
tPInt            aSz\
)\
{\
	bool verbose = 1;\
	if (verbose) std::cout<<"[IgnSocleImageLoader::LoadNCanaux] START"<<std::endl;\
	if (verbose) std::cout<<"[IgnSocleImageLoader::LoadNCanaux] lecture de ("<<aSz.real()<<", "<<aSz.imag()<<\
			") pixels dans "<<boost::filesystem::path(m_Nomfic).stem()<<" a dezoom "<<aDeZoom<<" a partir de ("<<aP0File.real()<<", "<<aP0File.imag()<<") et on le colle dans le buffer a ("<<aP0Im.real()<<", "<<aP0Im.imag()<<") "<<std::endl;\
	ign::image::io::ImageInput img(m_Nomfic);\
	if (!img.isValid()) return;\
	int lineOffset  =	aP0Im.imag();\
	int pixelOffset =	aP0Im.real();\
	int nPixelSpace = 1;\
	ign::image::eTypePixel typePixel = ign::image::TypePixel<aType>::GetType();\
	if (verbose) std::cout<<"type du buffer memoire: "<<ign::image::TypeToString(typePixel)<<std::endl;\
	for (size_t b = 0; b <aVImages.size(); ++b)\
	{\
		int nLineSpace = aVImages[b].mSzIm.real();\
		int nBandSpace = 1;\
		if (verbose) std::cout<<"canal "<<b<<": Sz = ("<< aVImages[b].mSzIm.real()<<", "<<aVImages[b].mSzIm.imag()<<")"<<std::endl;\
		int bdeOffset = 0;\
		size_t shift = (lineOffset * nLineSpace + nPixelSpace * pixelOffset + bdeOffset * nBandSpace)*ign::image::TypeSize(typePixel);\
		if (verbose) std::cout<<"on decale le pointeur de "<<shift<<" bits"<<std::endl;\
		char* charPtr = (char*)(aVImages[b].mDataLin);\
		void* ptr = (void*)(charPtr + shift);\
		if (verbose) std::cout<<"[IgnSocleImageLoader::LoadNCanaux] read"<<std::endl;\
		if (verbose) std::cout<<"[aVImages:]"<<nPixelSpace<<" "<<nLineSpace<<" "<<nBandSpace<<" | shift= "<<shift<<std::endl;\
		img.read(aP0File.real(), m_SzIm.imag() - aP0File.imag(), b, aSz.real()*aDeZoom, aSz.imag()*aDeZoom, 1, aDeZoom, ptr, typePixel, nPixelSpace,nLineSpace,nBandSpace);\
		if (verbose)\
		{\
			ign::image::BufferImage<aType> testBuf(aVImages[b].mSzIm.real(), aVImages[b].mSzIm.imag(), 1);\
			nPixelSpace = testBuf.getPixelSpace();\
			nLineSpace = testBuf.getLineSpace();\
			nBandSpace = testBuf.getBandSpace();\
			shift = (lineOffset * nLineSpace + nPixelSpace * pixelOffset + bdeOffset * nBandSpace)*ign::image::TypeSize(typePixel);\
			char* charPtr = (char*)(testBuf.getPtr());\
			void* ptr = (void*)(charPtr + shift);\
			if (verbose) std::cout<<"[Buff:]"<<nPixelSpace<<" "<<nLineSpace<<" "<<nBandSpace<<" | shift= "<<shift<<std::endl;\
			img.read(aP0File.real(), m_SzIm.imag() - aP0File.imag(), b, aSz.real()*aDeZoom, aSz.imag()*aDeZoom, 1, aDeZoom, ptr, typePixel, nPixelSpace,nLineSpace,nBandSpace);\
			std::ostringstream oss;\
			oss<<"/Temp/"<<boost::filesystem::path(m_Nomfic).stem().generic_string()<<"_testCrop"<<b<<".TIF";\
			testBuf.save(oss.str());\
			std::ostringstream oss2;\
			oss2<<"/Temp/"<<boost::filesystem::path(m_Nomfic).stem().generic_string()<<"_testELISE"<<b<<".TIF";\
			_debug_IO((const std::vector<sLowLevelIm<float> > &)aVImages, aSz, b, oss.str(), oss2.str());\
		}\
		if (verbose) std::cout<<"[IgnSocleImageLoader::LoadNCanaux] END"<<std::endl;\
	}\
}\


template <class Type,class TyBase> Im2D<Type,TyBase>::~Im2D()
{
}

namespace NS_ParamMICMAC
{	
	/*template <class aType>
	void _debug_IO(const std::vector<sLowLevelIm<aType> > & aVImages, cInterfModuleImageLoader::tPInt aSz, const std::string &filename)
	{
		Tiff_Im  aTOut( filename.c_str(), Pt2di(aSz.real(),aSz.imag()), GenIm::real4,
					   Tiff_Im::No_Compr,  Tiff_Im::BlackIsZero );
		
		Im2D<aType,double> aIm(aVImages[0].mDataLin,aVImages[0].mData,aSz.real(),aSz.imag());
		
		
		ELISE_COPY
		(
		 aTOut.all_pts(),
		 aIm.in(),
		 aTOut.out()
		 );
	}*/
	void _debug_IO(const std::vector<sLowLevelIm<float> > & aVImages, cInterfModuleImageLoader::tPInt aSz, int band, const std::string &filename, const std::string &filename2)
	{
		Tiff_Im  aTOut( filename2.c_str(), Pt2di(aSz.real(),aSz.imag()), GenIm::real4,
					   Tiff_Im::No_Compr,  Tiff_Im::BlackIsZero );
		
		Im2D<float,double> aIm(aVImages[band].mDataLin,aVImages[band].mData,aSz.real(),aSz.imag());
		
		
		ELISE_COPY
		(
		 aTOut.all_pts(),
		 aIm.in(),
		 aTOut.out()
		 );
		
		//check en relecture
		ign::image::BufferImage<float> img1(filename);
		ign::image::BufferImage<float> img2(filename2);
		for (int l = 0; l < aSz.imag(); ++l)
			for (int c  = 0; c <aSz.real(); ++c)
				if (img1.get(c,l,0) != img2.get(c,l,0))
				{
					std::ostringstream oss;
					oss<<"Incoherence dans "<<filename<<" au pixel: "<<c<<", "<<l;
					IGN_THROW_EXCEPTION(oss.str());
				}
		
		boost::filesystem::remove(filename);
		//boost::filesystem::remove(oss2.str());
	}
	
	///
	///
	///
	IgnSocleImageLoader::IgnSocleImageLoader(std::string const &nomfic)
	{
		m_Nomfic=nomfic;
		ign::image::io::ImageInput img(nomfic);
		if (!img.isValid())
		{
			std::cout<<"image invalide pour le socle IGN"<<std::endl;
		}
		m_Nbc = img.numBands();
		
		m_SzIm=std::complex<int>(img.numCols(),img.numLines());
		
		if (img.typePixel() == ign::numeric::eUnsignedChar) 
		{
			m_Type = eUnsignedChar; 
			m_S = false;
		}
		else if (img.typePixel() == ign::numeric::eSignedShort)
		{
			m_Type = eSignedShort;
			m_S = true;
		}
		else if (img.typePixel() == ign::numeric::eUnsignedShort) 
		{
			m_Type = eUnsignedShort;
			m_S = false;
		}
		else if (img.typePixel() == ign::numeric::eFloat) 
		{
			m_Type = eFloat;
			m_S = true;
		}
		else
		{
			m_Type = eFloat;
		}
	}	
	
	///
	///
	///
	IgnSocleImageLoader::~IgnSocleImageLoader()
	{
	}
	
	///
	///
	///
	eTypeNumerique IgnSocleImageLoader::PreferedTypeOfResol(int aDeZoom)const
	{
		return m_Type;
	}
	
	///
	///
	///
	std::complex<int> IgnSocleImageLoader::Sz(int aDeZoom)const
	{
		return std::complex<int>(m_SzIm.real()/aDeZoom,m_SzIm.imag()/aDeZoom);
	}   
	
	///
	///
	///            
	int IgnSocleImageLoader::NbCanaux()const
	{       
		return m_Nbc;
	}
	
	
	///
	///
	///
	void IgnSocleImageLoader::PreparePyram(int aDeZoom)
	{
	}
	
	///
	///
	///
	void IgnSocleImageLoader::LoadCanalCorrel
	(
	 const sLowLevelIm<float> & anIm,
	 int              aDeZoom,
	 tPInt            aP0Im,
	 tPInt            aP0File,
	 tPInt            aSz
	 )
	{
		try {
			
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
		catch (ign::Exception& e)
		{
			std::cout << " IgnSocleImageLoader::LoadCanalCorrel -- probleme : "<< boost::diagnostic_information(e) <<std::endl;
			IGN_THROW_EXCEPTION(boost::diagnostic_information(e));
		}
		catch (...)
		{
			std::cout<<"autre exception..."<<std::endl;
		}
	}
		
	
	
	///
	///
	///


	
DefineMembreLoadN(short int)
DefineMembreLoadN(unsigned char)
DefineMembreLoadN(unsigned short)
DefineMembreLoadN(float)
DefineMembreLoadN(bool)
	
		
};

#endif

