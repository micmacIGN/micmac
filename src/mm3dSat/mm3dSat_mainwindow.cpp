#include "mm3dSat_mainwindow.h"
#include "ui_mm3dSat_mainwindow.h"

#include <iostream>
#include <QFileDialog>
#include <QFileInfo>

#define VERSION 0.9

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    hide_coord_BoxTerr();

    ui->groupBox_MoreOpt->setVisible(false);
    ui->dSpin_ResolTer->setEnabled(false);
    ui->cBox_force_ResolTer->setChecked(false);
    ui->button_SelectMasq->setVisible(false);
    ui->label_or->setVisible(false);


    QObject::connect(ui->File_actionExit, SIGNAL(triggered()), this, SLOT(close()));
    QObject::connect(ui->File_actionReload, SIGNAL(triggered()), this, SLOT(reload_all_plus()));

    QObject::connect(ui->button_SelectImg, SIGNAL(clicked()), this, SLOT(reload_all()));
    QObject::connect(ui->button_SelectImg, SIGNAL(clicked()), this, SLOT(open_imFiles()));

    QObject::connect(ui->button_SelectMasq, SIGNAL(clicked()), this, SLOT(open_masqFile()));
    QObject::connect(ui->button_BoxTerr, SIGNAL(clicked()), this, SLOT(show_coord_BoxTerr()));

    QObject::connect(ui->button_moreOpt, SIGNAL(clicked()), this, SLOT(show_moreOptions()));

    QObject::connect(ui->cBox_force_ResolTer, SIGNAL(clicked()), this, SLOT(active_resolTerr()));

    QObject::connect(ui->button_startCompute, SIGNAL(clicked()), this, SLOT(start_work()));


    append_permanentText_to_textOutput();
    show_message("Please select image files");

    command_arg="";
    command="MICMAC ParamTer.xml";

    boxTerr_xmin=-10000;
    boxTerr_xmax=-10000;
    boxTerr_ymin=-10000;
    boxTerr_ymax=-10000;

    ui->groupBox_MMParams->setEnabled(false);
    ui->cBox_Exe->setEnabled(false);
    ui->button_startCompute->setEnabled(false);

}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::open_imFiles()
{
    //ui->button_SelectImg->setEnabled(false);

    QStringList img_list = QFileDialog::getOpenFileNames(this, "Choose Image Files", "/home","Images (*.tif *.tiff *.png *.jpg)");
    //std::cout<<"Open file: "<<fileNames[0].toStdString()<<std::endl;


    if (img_list.size()>1)
    {
        std::cout<<"Nb images: "<<img_list.size()<<std::endl;
        QFileInfo info(img_list.at(0));
        working_dir = info.absolutePath(); //returns a file's absolute path (the file name is not included)
        append_titleSec_to_textOutput("Working directory: ");
        append_to_textOutput(working_dir);
        append_titleSec_to_textOutput("Images and orientation grids: ");

        foreach (QString fileN, img_list)
        {
           if (fileN.size()!=0)
           {
               QFileInfo info(fileN);
               QString img_name=info.fileName(); //returns the name of the file, excluding the path
               img_names_list<<img_name;
               append_to_textOutput(img_name);
               QString base_name=info.baseName(); //returns the base name of the file without the path
               QString ori_name(base_name+".GRI");
               QDir qdir_working_dir(working_dir);
               if (qdir_working_dir.exists(ori_name))
               {
                   std::cout<<"Ori file OK: "<<ori_name.toStdString()<<std::endl;
                   ori_names_list<<ori_name;
                   append_to_textOutput(ori_name);
                   append_to_textOutput("   ");
                   ui->groupBox_MMParams->setEnabled(true);
                   ui->cBox_Exe->setEnabled(true);
                   ui->button_startCompute->setEnabled(true);
               }
               else
               {
                   QString error=QString("Error! The orientation grid ")+ori_name+QString(" does not exist!");
                   append_error_to_textOutput(error);
                   std::cout<<"Problem with ori file: "<<ori_name.toStdString()<<std::endl;
               }
          }
           else
           {
               QString error=QString("Error! This filename is not working: ")+fileN;
               append_error_to_textOutput(error);
           }
        }
    }
    else
     {
        append_error_to_textOutput("Insufficient number of images! There should be at least two!");
      }

    //ui->button_SelectImg->setText("Another image files");
}


void MainWindow::open_masqFile()
{
    masqFile = QFileDialog::getOpenFileName(this, tr("Open Masq File"),"/home",tr("Images (*.tif *.tiff *.png *.jpg)"));
    QFileInfo info(masqFile);

    append_titleSec_to_textOutput("Mask image file: ");
    append_to_textOutput(info.fileName());
   // append_titleSec_to_textOutput("Mask image directory: ");
    //append_to_textOutput(info.absolutePath());

    if (info.absolutePath()!=working_dir)
        append_error_to_textOutput("Error! Mask file should be in the same directory as images!");
    else
    {
        //std::cout<<info.suffix().toStdString()<<std::endl;
        if (info.suffix()!=("tif") && info.suffix()!=("tiff"))
            append_error_to_textOutput("Error! Mask file is not a TIFF file!");
        else
        {
            std::cout<<info.baseName().toStdString()<<std::endl;
            QString masq_xml=info.baseName()+".xml";
            std::cout<<masq_xml.toStdString()<<std::endl;
            QDir qdir_working_dir(working_dir);
            if (qdir_working_dir.exists(masq_xml))
                command_arg +=" +UseMasq=true +Masq=" + info.baseName();
            else
                append_error_to_textOutput("\n Error! Mask's xml file does not exist!");
        }
    }
}

void MainWindow::show_coord_BoxTerr()
{
    ui->label_xmin->setVisible(!ui->label_xmin->isVisible());
    ui->label_ymin->setVisible(!ui->label_ymin->isVisible());
    ui->label_xmax->setVisible(!ui->label_xmax->isVisible());
    ui->label_ymax->setVisible(!ui->label_ymax->isVisible());
    ui->spin_xmin->setVisible(!ui->spin_xmin->isVisible());
    ui->spin_ymin->setVisible(!ui->spin_ymin->isVisible());
    ui->spin_xmax->setVisible(!ui->spin_xmax->isVisible());
    ui->spin_ymax->setVisible(!ui->spin_ymax->isVisible());

    if (!ui->label_xmin->isVisible())
        ui->button_BoxTerr->setText("Set BoxTerrain");
    else
        ui->button_BoxTerr->setText("Disable BoxTerrain");
}

void MainWindow::hide_coord_BoxTerr()
{
    ui->label_xmin->setVisible(false);
    ui->label_ymin->setVisible(false);
    ui->label_xmax->setVisible(false);
    ui->label_ymax->setVisible(false);
    ui->spin_xmin->setVisible(false);
    ui->spin_ymin->setVisible(false);
    ui->spin_xmax->setVisible(false);
    ui->spin_ymax->setVisible(false);
}

void MainWindow::show_moreOptions()
{
    ui->groupBox_MoreOpt->setVisible(!ui->groupBox_MoreOpt->isVisible());
    if (!ui->groupBox_MoreOpt->isVisible())
        ui->button_moreOpt->setText("More options...");
    else
        ui->button_moreOpt->setText("Fewer options...");
}


void MainWindow::show_message(QString str)
{
    ui->statusBar->showMessage(str);
    append_to_textOutput(QString("<h3>")+str+QString("</h3>"));
}

void MainWindow::append_to_textOutput(QString str)
{
    ui->textOutput->append(str);
    this->repaint();
}

void MainWindow::append_titleSec_to_textOutput(QString str)
{
    ui->textOutput->append(QString("<h4>")+str+QString("</h4>"));
    this->repaint();
}

void MainWindow::append_error_to_textOutput(QString str)
{
    append_to_textOutput(QString("<h5><font color=\"blue\">")+str+QString("</font></h5>"));
    this->repaint();
}

void MainWindow::append_permanentText_to_textOutput()
{
    append_to_textOutput("***********************************************");
    show_message("MM3dSat v"+QString::number(VERSION));
    append_to_textOutput("***********************************************");
}

void MainWindow::define_BoxTerr()
{
    if ((ui->spin_xmin->value()!=0) || (ui->spin_ymin->value()!=0) || (ui->spin_xmax->value()!=0) || (ui->spin_ymax->value()!=0))
    {
        boxTerr_xmin=ui->spin_xmin->value();
        boxTerr_xmax=ui->spin_xmax->value();
        boxTerr_ymin=ui->spin_ymin->value();
        boxTerr_ymax=ui->spin_ymax->value();

        if ((ui->spin_xmax->value())<(ui->spin_xmin->value()))
        {
            //std::cout<<"User Xmin: "<<ui->spin_xmin->value()<<std::endl;
            //std::cout<<"User Xmax: "<<ui->spin_xmax->value()<<std::endl;
            boxTerr_xmin=ui->spin_xmax->value();
            boxTerr_xmax=ui->spin_xmin->value();

        }
        if ((ui->spin_ymax->value())<(ui->spin_ymin->value()))
        {
            boxTerr_ymin=ui->spin_ymax->value();
            boxTerr_ymax=ui->spin_ymin->value();
        }

        std::cout<<"Xmin: "<<boxTerr_xmin<<"    Ymin: "<<boxTerr_ymin<<"    Xmax: "<<boxTerr_xmax<<"    Ymax: "<<boxTerr_ymax<<std::endl;

       command_arg+=" +Xmin="+QString::number(boxTerr_xmin)
                        + " +Ymin="+QString::number(boxTerr_ymin)
                        + " +Xmax="+QString::number(boxTerr_xmax)
                        + " +Ymax="+QString::number(boxTerr_ymax);
        //std::cout<<command_arg.toStdString()<<std::endl;

    }
}


void MainWindow::define_mandatory_paramsValue()
{
    command_arg+=" +Inc="+QString::number(ui->dSpin_ZInc->value())
            + " +AltMoy="+QString::number(ui->dSpin_ZMoy->value())
            + " +SzW="+QString::number(ui->spin_SzW->value())
            + " +DilatAlti="+QString::number(ui->spin_ZDilatAlti->value())
            + " +DilatPlani="+QString::number(ui->spin_ZDilatPlani->value());
            //+ " +Regul0="+QString::number(ui->dSpin_ZRegul->value());

    //command+=command_arg;
}


void MainWindow::define_paramsValue_MoreOpt()
{
    command_arg+=" +RegulInit="+QString::number(ui->dSpin_ZRegulInit->value())
            +" +RegulFin="+QString::number(ui->dSpin_ZRegulFin->value())
            +" +PasInit="+QString::number(ui->dSpin_ZPasInit->value())
            +" +PasFin="+QString::number(ui->dSpin_ZPasFin->value());
            /*+" +ZoomInit= "+Qstring::number(ui->spin_DeZoomInit->value())
            +" +ZoomFin= "+Qstring::number(ui->spin_DeZoomFin->value());*/

    if (ui->cBox_force_ResolTer->isChecked() && ui->dSpin_ResolTer->value()!=0)
        command_arg+=" +UseResol=true +ResolTer=" +QString::number(ui->dSpin_ResolTer->value());


}

void MainWindow::active_resolTerr()
{
    if (ui->cBox_force_ResolTer->isChecked())
    {
        std::cout<<"cbox resol checked"<<std::endl;
        ui->dSpin_ResolTer->setEnabled(true);
    }
    else
    {
        std::cout<<"cbox resol unchecked"<<std::endl;
        ui->dSpin_ResolTer->setEnabled(false);
    }
}


int MainWindow::run_command(QString comm)
{
    int ret=0;
    std::cout<<"Try to run: "<<comm.toStdString()<<std::endl;
    append_to_textOutput(QString("Try to run: ")+comm);
    FILE *fp;
    char path[1035];
    fp = popen((comm+" 2>&1").toStdString().c_str(), "r");
    if (fp == NULL){
        std::cout<<"Failed to run command."<<std::endl;
       append_to_textOutput("Failed to run command.");
    }
    while (fgets(path, sizeof(path)-1, fp) != NULL){
        std::cout<<"Command output: "<<path<<std::endl;
       append_to_textOutput(QString("Command output: ")+path);
    }
    pclose(fp);
    return ret;
}


void MainWindow::start_work()
{
    ui->button_startCompute->setEnabled(false);

    append_to_textOutput("MicMac");

    if (ui->label_xmin->isVisible())
        define_BoxTerr();

    define_mandatory_paramsValue();

    if (ui->label_ZRegulInit->isVisible())
        define_paramsValue_MoreOpt();


    command+=command_arg;
    std::cout<<command.toStdString()<<std::endl;

    if (ui->cBox_Exe->isChecked())
    {
        run_command(command);
    }
    else
    {
        std::cout<<"Command not to be executed: "<<command.toStdString()<<std::endl;
        append_to_textOutput("Command not to be executed:\n"+command);
     }
}


void MainWindow::reload_all()
{
    //ui->button_SelectImg->setText("Select Image Files");

    hide_coord_BoxTerr();

    ui->groupBox_MMParams->setEnabled(false);

    ui->groupBox_MoreOpt->setVisible(false);
    ui->button_moreOpt->setText("More options...");
    ui->cBox_force_ResolTer->setChecked(false);
    ui->dSpin_ResolTer->setValue(0);

    ui->cBox_Exe->setChecked(false);

    ui->textOutput->clear();

    append_permanentText_to_textOutput();
    //show_message("Please select image files");

    ui->button_startCompute->setEnabled(false);
}

void MainWindow::reload_all_plus()
{
    reload_all();
    show_message("Please select image files");
}

