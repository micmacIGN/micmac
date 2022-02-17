#include "../uti_image/Sift/Sift.h"
#include "../uti_image/NewRechPH/cParamNewRechPH.h"
#include "../uti_image/NewRechPH/ExternNewRechPH.h"
#include "../TpMMPD/TpPPMD.h"


int ExtractSiftPt(std::string aNameIm, int aSzSift)
{

        std::string aOri;

        std::string aDir = DirOfFile(aNameIm);
        std::string afileName = NameWithoutDir(aNameIm);
        std::string aNameSift;

        getPastisGrayscaleFilename(aDir,afileName,aSzSift,aNameSift);

		if (aSzSift>0)
        	aNameSift  = DirOfFile(aNameSift) +"Pastis/" +  "LBPp" + NameWithoutDir(aNameSift) + ".dat";
		else
		{
			aNameSift  = "LBPp" + afileName + ".dat";
        	aNameSift = "Pastis/" + aNameSift;
		}


		std::cout << aNameSift << "\n";


        //read the file to know its original size
        Tiff_Im aFileInit = PastisTif(aNameIm);
        Pt2di       imageSize = aFileInit.sz();

        //calculate the scaling factor
        double aSSF =  (aSzSift<0) ? 1.0 :   double( ElMax( imageSize.x, imageSize.y ) ) / double( aSzSift ) ;

        //read key-pts
        std::vector<Siftator::SiftPoint> aVSift;
        bool Ok = read_siftPoint_list(aNameSift,aVSift);
        if(Ok == false)
		{
			std::cout << "something went wrong" << "\n";
            return 0;
		}

        std::string aNewDir = aDir + "unzip/";
        ELISE_fp::MkDir(aNewDir);
        aNewDir = aDir + "unzip/" + afileName + ".sift/";
        ELISE_fp::MkDir(aNewDir);


        //std::string aNameFile = aNameIm+".sift/keypoints.txt";
        std::string aNameFile = aDir + "unzip/" + afileName + ".sift/keypoints.txt";
        FILE * fpKeypoints = fopen(aNameFile.c_str(), "w");
        aNameFile = aDir + "unzip/" + afileName + ".sift/descriptors.txt";
        FILE * fpDescriptor = fopen(aNameFile.c_str(), "w");
        aNameFile = aDir + "unzip/" + afileName + ".sift/scores.txt";
        FILE * fpScores = fopen(aNameFile.c_str(), "w");
        aNameFile = aDir + "unzip/" + afileName + ".sift/otherInfo.txt";
        FILE * fpOther = fopen(aNameFile.c_str(), "w");
        int nSize = aVSift.size();
        for(int i=0; i<nSize; i++)
        {
            fprintf(fpKeypoints, "%lf %lf %lf\n", aVSift[i].x*aSSF, aVSift[i].y*aSSF, 1.0);
            fprintf(fpOther, "%lf %lf; %lf, %lf\n", aVSift[i].x*aSSF, aVSift[i].y*aSSF, aVSift[i].scale, aVSift[i].angle);
            fprintf(fpScores, "%lf\n", 100.0);
            for(int j=0; j<SIFT_DESCRIPTOR_SIZE; j++)
                fprintf(fpDescriptor, "%lf\t", aVSift[i].descriptor[j]);
            fprintf(fpDescriptor, "\n");
        }
        fclose(fpKeypoints);
        fclose(fpScores);
        fclose(fpDescriptor);
        fclose(fpOther);

        return EXIT_SUCCESS;
}

int TestLulin_main(int argc, char **argv)
{
    std::string aNameIm;
    int aSzSift;

    std::string aOri;

    ElInitArgMain
    (
        argc,argv,
        LArgMain() << EAMC(aNameIm,"Image Name (Dir+Pattern)")
                   << EAMC(aSzSift,"Resolution parameter"),
        LArgMain()
    );

    ExtractSiftPt(aNameIm, aSzSift);

    return EXIT_SUCCESS;
}
