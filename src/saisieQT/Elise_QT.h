#ifndef ELISE_QT_H
#define ELISE_QT_H

#include "general/CMake_defines.h"

#ifdef _WIN32
    #include "windows.h"
#endif

#if ELISE_Darwin
    #include <OpenGL/gl.h>
#else
    #include <GL/gl.h>
#endif

#ifdef Int
    #undef Int
#endif

#include <QAction>

#include <QApplication>
#include <QAbstractTableModel>
#include <QColor>
#include <QDebug>
#include <QDir>
#include <QDomDocument>
#include "qdesktopwidget.h"
#include <QFutureWatcher>
#include <QDialog>
#include <QDialogButtonBox>
#include <QFile>
#include <QFileDialog>

#include <QGLWidget>
#include <QGLShaderProgram>
#include <QGLContext>
#include <QGLBuffer>

#if ELISE_QT_VERSION == 5
#include <QOpenGLContext>
#endif


#include <QGridLayout>
#include <QtGui>
#include <QIcon>

#include <QImage>
#include <QImageReader>
#include <QInputDialog>
#include "qiodevice.h"
#include <QtGui/QMouseEvent>
#include <QMainWindow>
#include <QMenu>
#include <QMessageBox>
#include <QMimeData>
#include <QPainter>
#include <QPoint>
#include <QProgressDialog>
#include <QProcess>
#include <QTextStream>
#include <QTime>
#include <QTimer>
#include <QtConcurrentRun>
#include <QTableView>
#include <QTreeView>
#include <QSettings>
#include <QSignalMapper>
#include <QStandardItemModel>
#include <QStyle>
#include <QSortFilterProxyModel>
#include <QUrl>
#include <QVector>
#include <QWidget>


#endif // ELISE_QT_H
