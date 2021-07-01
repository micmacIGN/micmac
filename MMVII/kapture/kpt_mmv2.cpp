#include "include/MMVII_all.h"
#include "kapture.h"

namespace MMVII
{


/* ==================================================== */
/*                                                      */
/*          cAppli_Kapture*/
/*                                                      */
/* ==================================================== */


std::ostream& operator<<(std::ostream& os, const Kapture::QRot& rot)
{
    os << "[" << rot.w() << "," << rot.x() << "," << rot.y() << "," << rot.z()  << "]";
    return os;
}

std::ostream& operator<<(std::ostream& os, const Kapture::Vec3D& v)
{
    os << "[" << v.x() << "," << v.y() << "," << v.z()  << "]";
    return os;
}

std::ostream& operator<<(std::ostream& os, const Kapture::Orientation& o)
{
    if (o.hasRot())
        os << o.q();
    os << ":";
    if (o.hasVec())
        os << o.t();
    return os;
}

std::ostream& operator<<(std::ostream& os, const std::vector<double>& v)
{
    std::string sep="";
    os << "(";
    for (const auto& val: v) {
        std::cout << sep << val;
        sep = ",";
    }
    os << ")";
    return os;
}


class cAppli_Kapture : public cMMVII_Appli
{
     public :
        cAppli_Kapture(const std::vector<std::string> &  aVArgs,const cSpecMMVII_Appli &);
        int Exe() override;
        cCollecSpecArg2007 & ArgObl(cCollecSpecArg2007 & anArgObl) override;
        cCollecSpecArg2007 & ArgOpt(cCollecSpecArg2007 & anArgOpt) override;
     private :
        std::string mProjectPath;
        std::string mCommand;
        std::string mCmdArg1,mCmdArg2;
};


cCollecSpecArg2007 & cAppli_Kapture::ArgObl(cCollecSpecArg2007 & anArgObl)
{
   return
      anArgObl
          <<   Arg2007(mProjectPath,"Kapture set directory",{eTA2007::DirProject})
          <<   Arg2007(mCommand,"command: ",{})
   ;
}

cCollecSpecArg2007 & cAppli_Kapture::ArgOpt(cCollecSpecArg2007 & anArgOpt)
{
   return
      anArgOpt
          << AOpt2007(mCmdArg1,"R1","Filter/img1 (regex), def=''",{})
          << AOpt2007(mCmdArg2,"R2","Img2  (regex), def=''",{})
   ;
}




cAppli_Kapture::cAppli_Kapture(const std::vector<std::string> &  aVArgs,const cSpecMMVII_Appli & aSpec) :
  cMMVII_Appli (aVArgs,aSpec)
{

}

// Stupid example of custom Homologous point
struct Point {
    float x,y;
    float z;
    Point (float x, float y) : x(x), y(y),z(0) {}
};

struct Homol
{
    Point p1,p2;
    bool ok;
    Homol (float x1, float y1, float x2, float y2) : p1(x1,y1),p2(x2,y2) {}
};


int cAppli_Kapture::Exe()
{
    Kapture::setProject(mProjectPath);
    Kapture::Project &p = Kapture::project();

    std::cout << "Project version     : " << p.version() << "\n";
    std::cout << "Current API version : " << Kapture::Project::currentVersion() << "\n";

    p.load();

    if (mCommand == "save") {
        if (mCmdArg1 == "") {
            std::cerr  << "cmd 'save' needs a directory arg.\n";
            exit (1);
        }
        p.save(mCmdArg1);
    }

    if (mCommand == "cameras") {
        for (const auto& c : p.cameras()) {
            std::cout << c.sensor_device_id() << ":" << c.name() << ":" << c.modelStr() << ":";
            for (const auto& p : c.model_params())
                std::cout << p << ",";
            std::cout << "\n";
        }
        exit(EXIT_SUCCESS);
    }

    if (mCommand == "records") {
        for (const auto& record : p.imageRecords()) {
            std::cout << record.timestamp() << ":" << record.device_id() << ":" << record.image_path() << ":";
            if (record.trajectory())
                std::cout << *record.trajectory();
            else
                std::cout << ":";
            std::cout << ":";
            if (record.camera())
                std::cout << record.camera()->sensor_device_id() << ":" << record.camera()->model_params();
            else
                std::cout << ":";
            std::cout << "\n";
        }
        exit(EXIT_SUCCESS);
    }

    if (mCommand == "images") {
        for (const auto &i : p.imagesMatch(mCmdArg1))
            std::cout <<  i.string() << ":" << p.imagePath(i) << "\n";
        exit(EXIT_SUCCESS);
    }

    if (mCommand == "traj") {
        for (const auto &t : p.trajectories()) {
            std::cout << t.timestamp() << ":" << t.device_id() << ":" ;
            std::cout << t << "\n";
        }
        exit(EXIT_SUCCESS);
    }

    if (mCommand == "rigs") {
        for (const auto &r : p.rigs()) {
            std::cout << r.rig_device_id() << ":" << r.sensor_device_id() << ":" ;
            std::cout << r << "\n";
        }
        exit(EXIT_SUCCESS);
    }

    if (mCommand == "matches") {
        auto allMatches = p.allCoupleMatches(mCmdArg1);
        for (const auto&[img1, img2] : allMatches)
            std::cout <<  img1 << ":" << img2<< "\n";
        std::cout << allMatches.size() << " homologous files\n";
        exit(EXIT_SUCCESS);
    }

    if (mCommand == "match") {
        auto img1 = p.imageMatch(mCmdArg1);
        auto img2 = p.imageMatch(mCmdArg2);

//        auto myMatch = p.readMatches<Homol>(img1,img2);
//        for (const auto& m: myMatch)
//            std::cout << "(" << m.p1.x << "," << m.p1.y << "),(" << m.p2.x << "," << m.p2.y << ")\n";

//        auto matches = p.readMatches(img1,img2);
//        for (const auto& m: matches)
//            std::cout << "(" << m.x1 << "," << m.y1 << "),(" << m.x2 << "," << m.y2 << ")\n";

//        Kapture::MatchList m;
//        p.readMatches(img1,img2,m);

//        std::vector<Homol> myMatch;
//        p.readMatches(img1,img2,myMatch);
        exit (EXIT_SUCCESS);
    }

   return EXIT_SUCCESS;
}


tMMVII_UnikPApli Alloc_Kapture(const std::vector<std::string> &  aVArgs,const cSpecMMVII_Appli & aSpec)
{
   return tMMVII_UnikPApli(new cAppli_Kapture(aVArgs,aSpec));
}

cSpecMMVII_Appli  TheSpecKapture
(
     "Kapture",
      Alloc_Kapture,
      "This command is used to test kapture API",
      {eApF::Test},
      {eApDT::FileSys},
      {eApDT::None},
      __FILE__
);

};
