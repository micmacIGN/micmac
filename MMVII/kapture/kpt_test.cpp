#include <stdlib.h>
#include "kapture.h"

// **********************************************************************
// At this time, all errors throw en exception of type Kapture::Error() !
// **********************************************************************


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

void Usage();


int main(int argc, char *argv[])
{

    if (argc < 2)
        Usage();

    Kapture::Project p(argv[1]);

// or  Kapture::Project p;
//     p.setRoot(argv[1]);

// To use global singleton kapture project:
//    Kapture::Project &p = Kapture::project();
//    Kapture::setProject(argv[1]);

    std::cout << "Project version : " << p.version() << "\n";
    std::cout << "Current version : " << Kapture::Project::currentVersion() << "\n";

    if (argc < 3)
        exit(EXIT_SUCCESS);
    std::string cmd = argv[2];

    if (cmd == "sensors") {
        auto sensors = argc > 3 ? p.readSensors("",argv[3],"") : p.readSensors();
        for (const auto& s : sensors) {
            std::cout << s.deviceId() << ":" << s.name() << ":" << s.typeStr() << ":";
            for (const auto& p : s.params())
                std::cout << p << ",";
            std::cout << "\n";
        }
        exit(EXIT_SUCCESS);
    }

    if (cmd == "cameras") {
        auto cameras = argc > 3 ? p.readCameras("",argv[3],"") : p.readCameras();
        for (const auto& c : cameras) {
            std::cout << c.deviceId() << ":" << c.name() << ":" << c.modelStr() << ":";
            for (const auto& p : c.modelParams())
                std::cout << p << ",";
            std::cout << "\n";
        }
        exit(EXIT_SUCCESS);
    }

    if (cmd == "records") {
        auto records = argc > 3 ? p.readImageRecords("",argv[3]) : p.readImageRecords();
        for (const auto& record : records)
            std::cout << record.timestamp() << ":" << record.device() << ":" << record.image() << "\n";
        exit(EXIT_SUCCESS);
    }

    if (cmd == "images") {
        if (argc < 4)
            Usage();
        for (const auto &i : p.imagesMatch(argv[3]))
            std::cout <<  i.string().c_str() << " " << p.imagePath(i) << "\n";
        exit(EXIT_SUCCESS);
    }

    if (cmd == "match") {
        if (argc < 5)
            Usage();
        auto img1 = p.imageName(argv[3]);
        auto img2 = p.imageName(argv[4]);

        auto matches = p.readMatches(img1,img2);
        for (const auto& m: matches) {
            std::cout << "(" << m.x1 << "," << m.y1 << "),(" << m.x2 << "," << m.y2 << ")\n";
        }
        Kapture::MatchList m;
        p.readMatches(img2,img2,m);

        auto monMatch = p.readMatches<Homol>(img1,img2);

        std::vector<Homol> myMatch;
        p.readMatches(img1,img2,myMatch);
        exit (EXIT_SUCCESS);
    }

    Usage();
    return 0;
}

void Usage()
{
    std::cout << "kpttest <datapath>\n";
    std::cout << "kpttest <datapath> sensors [regex name]\n";
    std::cout << "kpttest <datapath> cameras [regex name]\n";
    std::cout << "kpttest <datapath> records [regex name]\n";
    std::cout << "kpttest <datapath> images <regex>\n";
    std::cout << "kpttest <datapath> match <regex> <regex>\n";
    exit(1);
}

