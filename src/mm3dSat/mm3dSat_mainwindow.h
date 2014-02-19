#ifndef MM3DSAT_MAINWINDOW_H
#define MM3DSAT_MAINWINDOW_H

#include <QMainWindow>

namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT
    
public:
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();
    void show_message(QString str);
    void append_to_textOutput(QString str);
    void append_titleSec_to_textOutput(QString str);
    void append_error_to_textOutput(QString str);
    void append_permanentText_to_textOutput();
    void define_BoxTerr();
    void define_paramsValue_MoreOpt();
    void define_mandatory_paramsValue();


public slots:
    void open_imFiles();
    void open_masqFile();
    void show_coord_BoxTerr();
    void hide_coord_BoxTerr();
    void show_moreOptions();
    void start_work();
    int run_command(QString command);
    void reload_all();
    void reload_all_plus();
    void active_resolTerr();

    
private:
    Ui::MainWindow *ui;
    QString working_dir;
    QStringList img_names_list;
    QStringList ori_names_list;
    QString masqFile;
    QString command_arg;
    QString command;
    int boxTerr_xmin;
    int boxTerr_xmax;
    int boxTerr_ymin;
    int boxTerr_ymax;
};

#endif //  MM3DSAT_MAINWINDOW_H
