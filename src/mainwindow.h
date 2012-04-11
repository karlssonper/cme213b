#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QDebug>
#include <QMessageBox>
#include <QSqlDatabase>
#include <QSqlQuery>
#include <QSqlTableModel>
#include <QFile>

#include  <QHash>

#include <QDomDocument>
#include <QXmlDefaultHandler>
#include <iostream>

#include <QStringList>

 #include <QTreeWidgetItem>
#include <QTreeWidget>

namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT
    
public:
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();

    bool findMatchingData;

    // Given a list of models and corresponding algorithm name
    // find all common parameters
    QStringList findCommonParameters(QStringList models, QString algoName);
    
    // Given a model and parameter(s), find all other models that have the
    // same parameter
    QStringList findCommonModels(QString model, QString algoName,
                                 QStringList parameterList);

protected:
    bool createDatabase();
    bool connectToDataBase();
    bool addData(QDomDocument &Parameters, QString propertyName);

    QSqlTableModel* model;
    QSqlDatabase db;
    QHash<QString,int> propertyIDHash;

    // This stores a stringlist for each algorithm type that contains
    // the names of the parameters used for that algorithm
    QHash<QString,QStringList> parameterNameHashTable;

    void traverseNode(QDomNode node, QString path, QString *values,
                      QStringList *parameterList,
                      bool initial);

    void setupModel();

    int idCount;


private:
    Ui::MainWindow *ui;

};

class parameterNode: public QTreeWidgetItem
{

public:
    parameterNode(QString nodeName, QString nodeValue,
                  QTreeWidgetItem *parentItem = 0):
                  QTreeWidgetItem(parentItem,QTreeWidgetItem::UserType)
    {
        this->name_ = nodeName;
        this->value_ = nodeValue;
        this->setData(0,Qt::DisplayRole,this->name_);
        this->setData(1,Qt::DisplayRole,this->value_);
    }

    QHash<QString, parameterNode*> children;
    parameterNode* parent;

    QString name(){return name_;}
    QString value(){return value_;}

private:
    QString name_;
    QString value_;
};

class parameterModel : public QTreeWidget
{

public:
    parameterModel(QWidget *parent = 0);
    void addNode(QString pathName, QString value, parameterNode *parentNode);
    parameterNode *root(){return root_;}

private:
    parameterNode *root_;
};

#endif // MAINWINDOW_H
