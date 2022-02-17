#ifndef WORKBENCHWIDGET_H
#define WORKBENCHWIDGET_H

#include "Engine.h"

#include <QWidget>
#include <QPushButton>
#include <QFileDialog>
#include <QDebug>
#include <QLineEdit>
#include <QComboBox>
#include <QListView>
#include <QFileSystemModel>
#include <QTextEdit>

namespace Ui {
class cWorkBenchWidget;
}

class cWorkBenchWidget : public QWidget
{
	Q_OBJECT

public:
	explicit cWorkBenchWidget(QWidget *parent = 0);
	~cWorkBenchWidget();

	deviceIOCamera* dIOCamera() const;
	void			setDIOCamera(deviceIOCamera* dIOCamera);

	deviceIOTieFile* dIOTieFile() const;
	void setDIOTieFile(deviceIOTieFile* dIOTieFile);

	deviceIOImage* dIOImage() const;
	void setDIOImage(deviceIOImage* dIOImage);

protected:

	void		initModelFileImages();

	QComboBox*	comboB_Orientations();

	QListView*	listViewImages();

	QLineEdit*	lineImages();

	QLineEdit*	lineMainDir();

	QTextEdit*  textEditOrient();

	QLabel*		labeImage();

protected slots:

	void		chooseImages();

	void		updateOrientation(const QModelIndex& index);

	void		updateTiePoint(const QModelIndex& index);
private:

	Ui::cWorkBenchWidget *ui;

	QStringList			_filesImages;

	QDir				_mainDir;

	QStringList			_oriDirectory;

	QFileSystemModel*	_modelFileImage;
	QFileSystemModel*	_modelFileTie;

	QStringList			_filters_DIR_Ori;

	deviceIOCamera*		_dIOCamera;
	deviceIOTieFile*	_dIOTieFile;
	deviceIOImage*		_dIOImage;
};

#endif // WORKBENCHWIDGET_H
