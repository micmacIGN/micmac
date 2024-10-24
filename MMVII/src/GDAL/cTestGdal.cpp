#include "cMMVII_Appli.h"
#include <iostream>

#include "MMVII_PCSens.h"
#include "MMVII_Image2D.h"
#include "MMVII_enums.h"


namespace MMVII
{


    class cAppli_TestGdal : public cMMVII_Appli
    {
        public :

            cAppli_TestGdal(const std::vector<std::string> &  aVArgs,const cSpecMMVII_Appli &);
            int Exe() override;
            cCollecSpecArg2007 & ArgObl(cCollecSpecArg2007 & anArgObl) override;
            cCollecSpecArg2007 & ArgOpt(cCollecSpecArg2007 & anArgOpt) override;
            void OpenTiffImage();
            void OpenTiffImageCIm2D();
            void OpenTiffImageCIm2DBox();
            void OpenTiffRGBImageCIm2DBox();
            void CreateTiffImage();
            void WriteImage();
            void WriteRGBImage();
            void WriteImageToFile();
            void WriteRGBImageToFile();
            void WriteImageTiles();

        private :
            cPhotogrammetricProject  mPhProj;
            std::string              mImagePathBW;
            std::string              mImagePathRGB;
        
    };


    cAppli_TestGdal::cAppli_TestGdal
    (
        const std::vector<std::string> &  aVArgs,
        const cSpecMMVII_Appli & aSpec
    ) :
        cMMVII_Appli  (aVArgs,aSpec),
        mPhProj       (*this)
    {
    }



    cCollecSpecArg2007 & cAppli_TestGdal::ArgObl(cCollecSpecArg2007 & anArgObl)
    {
        return anArgObl
                <<  Arg2007(mImagePathBW,"Image BW")
                <<  Arg2007(mImagePathRGB,"Image RGB")
            ;
    }


    cCollecSpecArg2007 & cAppli_TestGdal::ArgOpt(cCollecSpecArg2007 & anArgOpt)
    {

        return    anArgOpt
        ;
    }


    void cAppli_TestGdal::OpenTiffImage()
    {
        StdOut() << "Ouverture de l'image : " << mImagePathBW << std::endl;
        cDataFileIm2D aImage = cDataFileIm2D::Create(mImagePathBW, eForceGray::No);
        StdOut() << "cDataFileIm2D creee" << std::endl;
        StdOut() << "Taille de l'image : " << aImage.Sz() << std::endl;
        StdOut() << "NbChannel : " << aImage.NbChannel() << std::endl;
        //StdOut() << "Type : " << aImage.Type() << std::endl;
        StdOut() << "Name : " << aImage.Name() << std::endl;
        StdOut() << "IsEmpty : " << aImage.IsEmpty() << std::endl;
        StdOut() << "" << std::endl;
    } 

    void cAppli_TestGdal::OpenTiffImageCIm2D()
    {
        StdOut() << "Ouverture de l'image : " << mImagePathBW << std::endl;
        cIm2D<tU_INT1> aIm2D = cIm2D<tU_INT1>::FromFile(mImagePathBW);
        cDataIm2D<tU_INT1> & aDataIm2D = aIm2D.DIm();
        StdOut() << "Taille de l'image : " << aDataIm2D.Sz() << std::endl;
        StdOut() << "Valeur du pixel (5000, 8000) : " << (int) aDataIm2D.GetV(cPt2di(5000, 8000)) << std::endl;
        StdOut() << "" << std::endl;

    }

    void cAppli_TestGdal::OpenTiffImageCIm2DBox()
    {
        StdOut() << "Ouverture de l'image avec une boite de (6000, 8000, 9000, 10000): " << mImagePathBW << std::endl;
        cBox2di aBox = cBox2di(cPt2di(6000, 8000), cPt2di(9000, 10000));
        cIm2D<tU_INT1> aIm2D = cIm2D<tU_INT1>::FromFile(mImagePathBW, aBox);
        StdOut() << "cDataFileIm2D créée" << std::endl;
        cDataIm2D<tU_INT1> & aDataIm2D = aIm2D.DIm();
        StdOut() << "Taille de l'image : " << aDataIm2D.Sz() << std::endl;
        StdOut() << "Valeur du pixel (0, 0) : " << aDataIm2D.GetV(cPt2di(0, 0)) << std::endl;
        StdOut() << "Valeur du pixel (0, 1) : " << aDataIm2D.GetV(cPt2di(0, 1)) << std::endl;
        StdOut() << "Valeur du pixel (1, 1) : " << aDataIm2D.GetV(cPt2di(1, 1)) << std::endl;
        StdOut() << "Valeur du pixel (1, 0) : " << aDataIm2D.GetV(cPt2di(1, 0)) << std::endl;
        StdOut() << "Valeur du pixel (0.5, 0.5) : " << aDataIm2D.DefGetVBL(cPt2dr(0.5, 0.5), -1) << std::endl;
        StdOut() << "Valeur du pixel (1000.5, 1000.5) : " << aDataIm2D.DefGetVBL(cPt2dr(1000.5, 1000.5), -1) << std::endl;

    } 


    void cAppli_TestGdal::OpenTiffRGBImageCIm2DBox()
    {
        StdOut() << "Ouverture de l'image RGB avec une boite de (6000, 8000, 9000, 10000): " << mImagePathRGB << std::endl;
        cBox2di aBox = cBox2di(cPt2di(6000, 8000), cPt2di(9000, 10000));
        cRGBImage aRGBIm = cRGBImage::FromFile(mImagePathRGB, aBox);
        StdOut() << "Valeur du pixel (0, 1) : " << aRGBIm.GetRGBPix(cPt2di(0, 1)) << std::endl;
        StdOut() << "Valeur du pixel (0.5, 0.5) : " << aRGBIm.GetRGBPixBL(cPt2dr(0.5, 0.5)) << std::endl;
    }


    void cAppli_TestGdal::CreateTiffImage()
    {
        std::string aCreatedImage = "image0.tif";
        StdOut() << "Ouverture de l'image : " << aCreatedImage << std::endl;
        cDataFileIm2D aImage = cDataFileIm2D::Create(aCreatedImage, eTyNums::eTN_INT2, cPt2di(1000, 1000), 3);
        StdOut() << "cDataFileIm2D creee : tINT2, 3 canaux" << std::endl;
        StdOut() << "Taille de l'image : " << aImage.Sz() << std::endl;
        StdOut() << "NbChannel : " << aImage.NbChannel() << std::endl;
        StdOut() << "Name : " << aImage.Name() << std::endl;
        StdOut() << "IsEmpty : " << aImage.IsEmpty() << std::endl;
        StdOut() << "" << std::endl;
    }



    void cAppli_TestGdal::WriteImage()
    {
        // Open image
        cBox2di aBox = cBox2di(cPt2di(7000, 8000), cPt2di(7100, 8100));
        cIm2D<tU_INT1> aIm2D = cIm2D<tU_INT1>::FromFile(mImagePathBW, aBox);
        // Create the new image of type eTyNums
        std::string aCreatedImageName = "image1.tif";
        cDataFileIm2D aCreatedImage = cDataFileIm2D::Create(aCreatedImageName, eTyNums::eTN_REAL8, cPt2di(100, 100), 1);
        // Open the new image
        // Data will be written in Type (tREAL4), but the image will be save in eTyNums (eTyNums::eTN_REAL8)
        cIm2D<tREAL4> aCreatedIm2D = cIm2D<tREAL4>::FromFile(aCreatedImageName);

        // Copy data
        for (int x = aCreatedIm2D.DIm().X0(); x < aCreatedIm2D.DIm().X1(); x++)
        {
            for (int y = aCreatedIm2D.DIm().Y0(); y < aCreatedIm2D.DIm().Y1(); y++)
            {
                aCreatedIm2D.DIm().SetV(cPt2di(x, y), aIm2D.DIm().GetV(cPt2di(x, y)));
            }
        }
        // Save image
        aCreatedIm2D.Write(aCreatedImage, cPt2di(0, 0));
    }

    void cAppli_TestGdal::WriteRGBImage()
    {
        // Open image in ((7000, 8000), (7100, 8100))
        cBox2di aBox = cBox2di(cPt2di(7000, 8000), cPt2di(7100, 8100));
        cRGBImage aRGBImage = cRGBImage::FromFile(mImagePathRGB, aBox);
        
        // Create the new image of size (200, 200). Type must be eTyNums::eTN_U_INT1 because it is RGB image
        std::string aCreatedImageName = "image2.tif";
        cDataFileIm2D aCreatedImage = cDataFileIm2D::Create(aCreatedImageName, eTyNums::eTN_U_INT1, cPt2di(200, 200), 3);
        
        // Open the new image in ((50, 50), (150, 150))
        // Origin of the box does not matter in writting because it is in Write method that the origin is defined. Only the size of the box is important here
        cRGBImage aCreatedIm2D = cRGBImage::FromFile(aCreatedImageName, cBox2di(cPt2di(50, 50), cPt2di(150, 150)));

        // Copy data
        for (int x = aRGBImage.ImR().DIm().X0(); x < aRGBImage.ImR().DIm().X1(); x++)
        {
            for (int y = aRGBImage.ImR().DIm().Y0(); y < aRGBImage.ImR().DIm().Y1(); y++)
            {
                aCreatedIm2D.SetRGBPix(cPt2di(x, y), aRGBImage.GetRGBPix(cPt2di(x, y)));
            }
        }

        // Save data in coordinates (50, 50)
        aCreatedIm2D.Write(aCreatedImage, cPt2di(50, 50));
    }


    void cAppli_TestGdal::WriteImageToFile()
    {
        // Open image in ((7000, 8000), (10000, 11000))
        cBox2di aBox = cBox2di(cPt2di(7000, 8000), cPt2di(10000, 11000));
        cIm2D<tU_INT1> aIm2D = cIm2D<tU_INT1>::FromFile(mImagePathBW, aBox);

        // Save extracted image   
        //aIm2D.DIm().ToFile(mImageCreatePath);

        // Save an extraction of the extracted image : ((8000, 9000), (9000, 10000)) of the first image
        std::string aCreatedImageName = "image3.tif";
        aIm2D.DIm().ClipToFile(aCreatedImageName, cRect2(cPt2di(1000, 1000), cPt2di(2000, 2000)));

    }


    void cAppli_TestGdal::WriteRGBImageToFile()
    {
        // Open image in ((7000, 8000), (10000, 11000))
        cBox2di aBox = cBox2di(cPt2di(7000, 8000), cPt2di(10000, 11000));
        cRGBImage aRGBImage = cRGBImage::FromFile(mImagePathRGB, aBox);

        // Read Image in ((15000, 12000), (18000, 15000)). Then aBox in aRGBImage is useless
        aRGBImage.Read(mImagePathRGB, cPt2di(0, 0), 1.0, cBox2di(cPt2di(15000, 12000), cPt2di(18000, 15000)));
        
        
        // Create new image
        std::string aCreatedImageName = "image4.tif";
        cDataFileIm2D aRGBImagecreated = cDataFileIm2D::Create(aCreatedImageName, eTyNums::eTN_U_INT1, cPt2di(3000, 3000), 3);
        // Write data
        aRGBImage.Write(aRGBImagecreated, cPt2di(0, 0));
        
        // Save extracted image in RGB
        //aRGBImage.ImR().DIm().ToFile(mImageCreatePath, aRGBImage.ImG().DIm(), aRGBImage.ImB().DIm());

    }

    void cAppli_TestGdal::WriteImageTiles()
    {
        // Open image in ((7000, 8000), (10000, 11000))
        cBox2di aBox = cBox2di(cPt2di(7000, 8000), cPt2di(10000, 11000));
        cIm2D<tU_INT1> aIm2D = cIm2D<tU_INT1>::FromFile(mImagePathBW, aBox);
        // Create the new image of size (3000, 3000)
        std::string aCreatedImageName = "image5.tif";
        cDataFileIm2D aCreatedImage = cDataFileIm2D::Create(aCreatedImageName, eTyNums::eTN_REAL8, cPt2di(3000, 3000), 1);
        // Open the new image with a size of (1000, 1000)
        // Origin of the box does not matter in writting because it is in Write method that the origin is defined. Only the size of the box is important here
        cIm2D<tREAL4> aTile = cIm2D<tREAL4>::FromFile(aCreatedImageName, cBox2di(cPt2di(0, 0), cPt2di(1000, 1000)));

        // Give the same result :
        // cIm2D<tREAL4> aTile = cIm2D<tREAL4>::FromFile(mImageCreatePath, cBox2di(cPt2di(300, 300), cPt2di(1300, 1300)));
        
        // Copy data
        for (int x = aTile.DIm().X0(); x < aTile.DIm().X1(); x++)
        {
            for (int y = aTile.DIm().Y0(); y < aTile.DIm().Y1(); y++)
            {
                aTile.DIm().SetV(cPt2di(x, y), aIm2D.DIm().GetV(cPt2di(x, y)));
            }
        }

        // Save image at the origin (0, 0)
        aTile.Write(aCreatedImage, cPt2di(0, 0));
        // Save the same image but at the origin (1000, 2000)
        aTile.Write(aCreatedImage, cPt2di(1000, 2000));
    }

    int cAppli_TestGdal::Exe()
    {
        mPhProj.FinishInit();  // the final construction of  photogrammetric project manager can only be done now

        OpenTiffImage();
        OpenTiffImageCIm2D();
        OpenTiffImageCIm2DBox();
        OpenTiffRGBImageCIm2DBox();
        CreateTiffImage();
        WriteImage();
        WriteRGBImage();
        WriteImageToFile();
        WriteRGBImageToFile();
        WriteImageTiles();

        return EXIT_SUCCESS;
    }                                       

    /* ==================================================== */
    /*                                                      */
    /*               MMVII                                  */
    /*                                                      */
    /* ==================================================== */


    tMMVII_UnikPApli Alloc_TestGdal(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec)
    {
    return tMMVII_UnikPApli(new cAppli_TestGdal(aVArgs,aSpec));
    }

    cSpecMMVII_Appli  TheSpec_TestGdal
    (
        "TestGdal",
        Alloc_TestGdal,
        "Test avec GDAL",
        {eApF::GCP,eApF::Ori},
        {eApDT::Orient},
        {eApDT::Xml},
        __FILE__
    );

}
