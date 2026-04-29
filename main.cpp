// ============================================================
// Proyecto Final — Redes Neuronales
// Benchmarking Regresión Logística: C++ con Eigen
// Universidad Sergio Arboleda
// Integrantes: Julian Rincón, Miguel Flechas, Andrés Castro, Juan Hurtado
// ============================================================

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cmath>
#include <chrono>
#include <algorithm>
#include <numeric>
#include <random>
#include <iomanip>

// Eigen — álgebra lineal de alto rendimiento
#include <Eigen/Dense>

using namespace Eigen;
using namespace std;
using namespace std::chrono;

// ============================================================
// TIPOS GLOBALES
// ============================================================
typedef MatrixXf Matrix_f;
typedef VectorXf Vector_f;
typedef VectorXi Vector_i;

// ============================================================
// CLASE: DataLoader
// Responsabilidad: Cargar dataset CSV preprocesado (MNIST binario 3 vs 8)
// ============================================================
class DataLoader {
public:
    // Carga un CSV con formato: label, pixel0, pixel1, ..., pixel783
    static bool loadCSV(const string& filepath,
                        Matrix_f& X,
                        Vector_i& y,
                        int max_rows = -1)
    {
        ifstream file(filepath);
        if (!file.is_open()) {
            cerr << "[ERROR] No se pudo abrir el archivo: " << filepath << endl;
            return false;
        }

        vector<vector<float>> X_data;
        vector<int>           y_data;
        string line;
        bool first_line = true;

        while (getline(file, line)) {
            if (first_line) { first_line = false; continue; } // saltar header
            if (max_rows > 0 && (int)X_data.size() >= max_rows) break;

            istringstream ss(line);
            string token;
            vector<float> row;
            int label = -1;
            int col = 0;

            while (getline(ss, token, ',')) {
                if (col == 0) {
                    label = stoi(token);
                } else {
                    row.push_back(stof(token));
                }
                col++;
            }

            if (label == 3 || label == 8) {
                y_data.push_back(label == 8 ? 1 : 0);
                X_data.push_back(row);
            }
        }

        // Convertir a Eigen matrices
        int n = (int)X_data.size();
        int d = (int)X_data[0].size();
        X.resize(n, d);
        y.resize(n);

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < d; j++) {
                X(i, j) = X_data[i][j];
            }
            y(i) = y_data[i];
        }

        return true;
    }

    // Normalización: StandardScaler (media=0, std=1) — localidad temporal en columnas
    static void standardScale(Matrix_f& X_train,
                               Matrix_f& X_test,
                               Vector_f& mean_out,
                               Vector_f& std_out)
    {
        int n = X_train.rows();
        mean_out = X_train.colwise().mean();
        std_out  = ((X_train.rowwise() - mean_out.transpose())
                        .array().square().colwise().sum() / n)
                        .sqrt().matrix();

        // Evitar división por cero
        std_out = std_out.array().max(1e-8f).matrix();

        X_train = (X_train.rowwise() - mean_out.transpose())
                      .array().rowwise() / std_out.transpose().array();
        X_test  = (X_test.rowwise() - mean_out.transpose())
                      .array().rowwise() / std_out.transpose().array();
    }

    // Split estratificado (train/test)
    static void trainTestSplit(const Matrix_f& X,
                               const Vector_i& y,
                               Matrix_f& X_train, Vector_i& y_train,
                               Matrix_f& X_test,  Vector_i& y_test,
                               float test_ratio = 0.2f,
                               int seed = 42)
    {
        int n = X.rows();
        vector<int> idx(n);
        iota(idx.begin(), idx.end(), 0);

        mt19937 rng(seed);
        shuffle(idx.begin(), idx.end(), rng);

        int test_size  = (int)(n * test_ratio);
        int train_size = n - test_size;

        X_train.resize(train_size, X.cols());
        y_train.resize(train_size);
        X_test.resize(test_size, X.cols());
        y_test.resize(test_size);

        for (int i = 0; i < train_size; i++) {
            X_train.row(i) = X.row(idx[i]);
            y_train(i)     = y(idx[i]);
        }
        for (int i = 0; i < test_size; i++) {
            X_test.row(i) = X.row(idx[train_size + i]);
            y_test(i)     = y(idx[train_size + i]);
        }
    }
};

// ============================================================
// CLASE: LogisticRegressionEigen
// Motor matemático: Sigmoide + Gradiente Descendente con Eigen
// ============================================================
class LogisticRegressionEigen {
private:
    Vector_f weights_;      // pesos w (784 dimensiones)
    float    bias_;         // sesgo b
    float    learning_rate_;
    int      epochs_;
    float    lambda_;       // regularización L2

    // Historial de pérdida para diagnóstico
    vector<float> loss_history_;

    // -------------------------------------------------------
    // Función Sigmoide: σ(z) = 1 / (1 + e^{-z})
    // Implementada sobre vectores Eigen para aprovechar SIMD
    // -------------------------------------------------------
    inline Vector_f sigmoid(const Vector_f& z) const {
        return (1.0f / (1.0f + (-z.array()).exp())).matrix();
    }

    // -------------------------------------------------------
    // Función de pérdida: Binary Cross-Entropy + L2
    // L = -1/n * Σ [ y·log(ŷ) + (1-y)·log(1-ŷ) ] + λ/2 * ||w||²
    // -------------------------------------------------------
    float computeLoss(const Vector_f& y_pred,
                      const Vector_i& y_true) const
    {
        int n = y_true.size();
        Vector_f y_true_f = y_true.cast<float>();
        float eps = 1e-9f;

        Vector_f log_pred     = (y_pred.array() + eps).log().matrix();
        Vector_f log_pred_inv = (1.0f - y_pred.array() + eps).log().matrix();

        float cross_entropy = -(y_true_f.dot(log_pred) +
                                (1.0f - y_true_f.array()).matrix().dot(log_pred_inv)) / n;
        float l2_reg = (lambda_ / 2.0f) * weights_.squaredNorm();

        return cross_entropy + l2_reg;
    }

public:
    // -------------------------------------------------------
    // Constructor
    // -------------------------------------------------------
    LogisticRegressionEigen(float lr     = 0.1f,
                            int   epochs = 500,
                            float lambda = 0.001f)
        : learning_rate_(lr), epochs_(epochs), lambda_(lambda), bias_(0.0f)
    {}

    // -------------------------------------------------------
    // Entrenamiento: Gradient Descent con Eigen
    // Gradiente: ∂L/∂w = 1/n * Xᵀ(ŷ - y) + λ·w
    //            ∂L/∂b = 1/n * Σ(ŷ - y)
    // -------------------------------------------------------
    void fit(const Matrix_f& X_train, const Vector_i& y_train) {
        int n = X_train.rows();
        int d = X_train.cols();

        // Inicialización Xavier
        mt19937 rng(42);
        normal_distribution<float> dist(0.0f, sqrt(2.0f / d));
        weights_ = Vector_f::Zero(d).unaryExpr([&](float) { return dist(rng); });
        bias_    = 0.0f;
        loss_history_.clear();

        Vector_f y_true_f = y_train.cast<float>();

        for (int epoch = 0; epoch < epochs_; epoch++) {
            // Forward pass: z = X·w + b,  ŷ = σ(z)
            Vector_f z      = X_train * weights_ + Vector_f::Constant(n, bias_);
            Vector_f y_pred = sigmoid(z);

            // Error
            Vector_f error = y_pred - y_true_f;

            // Gradientes
            Vector_f grad_w = (X_train.transpose() * error) / n
                              + lambda_ * weights_;
            float    grad_b = error.sum() / n;

            // Actualización de parámetros (Gradient Descent)
            weights_ -= learning_rate_ * grad_w;
            bias_    -= learning_rate_ * grad_b;

            // Registrar pérdida cada 50 épocas
            if (epoch % 50 == 0 || epoch == epochs_ - 1) {
                float loss = computeLoss(y_pred, y_train);
                loss_history_.push_back(loss);
                cout << "  Epoch " << setw(4) << epoch
                     << " | Loss: " << fixed << setprecision(6) << loss << endl;
            }
        }
    }

    // -------------------------------------------------------
    // Predicción de probabilidades
    // -------------------------------------------------------
    Vector_f predictProba(const Matrix_f& X) const {
        int n = X.rows();
        Vector_f z = X * weights_ + Vector_f::Constant(n, bias_);
        return sigmoid(z);
    }

    // -------------------------------------------------------
    // Predicción de clases (umbral 0.5)
    // -------------------------------------------------------
    Vector_i predict(const Matrix_f& X) const {
        Vector_f proba = predictProba(X);
        return (proba.array() >= 0.5f).cast<int>();
    }

    const vector<float>& lossHistory()  const { return loss_history_; }
    const Vector_f&      getWeights()   const { return weights_; }
    float                getBias()      const { return bias_; }
};

// ============================================================
// CLASE: Evaluator
// Calcula métricas de clasificación binaria
// ============================================================
class Evaluator {
public:
    struct Metrics {
        float accuracy;
        float precision;
        float recall;
        float f1_score;
        int   tp, tn, fp, fn;
    };

    static Metrics evaluate(const Vector_i& y_true,
                             const Vector_i& y_pred)
    {
        Metrics m = {0, 0, 0, 0, 0, 0, 0, 0};
        int n = y_true.size();

        for (int i = 0; i < n; i++) {
            if (y_true(i) == 1 && y_pred(i) == 1) m.tp++;
            else if (y_true(i) == 0 && y_pred(i) == 0) m.tn++;
            else if (y_true(i) == 0 && y_pred(i) == 1) m.fp++;
            else if (y_true(i) == 1 && y_pred(i) == 0) m.fn++;
        }

        m.accuracy  = (float)(m.tp + m.tn) / n;
        m.precision = (m.tp + m.fp > 0) ? (float)m.tp / (m.tp + m.fp) : 0.0f;
        m.recall    = (m.tp + m.fn > 0) ? (float)m.tp / (m.tp + m.fn) : 0.0f;
        m.f1_score  = (m.precision + m.recall > 0)
                      ? 2.0f * m.precision * m.recall / (m.precision + m.recall)
                      : 0.0f;
        return m;
    }

    static void printReport(const Metrics& m, double train_time_sec) {
        cout << "\n" << string(52, '=') << endl;
        cout << "  MÉTRICAS — C++ (Eigen) Logistic Regression" << endl;
        cout << string(52, '=') << endl;
        cout << fixed << setprecision(4);
        cout << "  Accuracy   : " << m.accuracy   * 100 << " %" << endl;
        cout << "  Precision  : " << m.precision  * 100 << " %" << endl;
        cout << "  Recall     : " << m.recall     * 100 << " %" << endl;
        cout << "  F1-Score   : " << m.f1_score   * 100 << " %" << endl;
        cout << string(52, '-') << endl;
        cout << "  Matriz de Confusión:" << endl;
        cout << "             Pred 3   Pred 8" << endl;
        cout << "  Real 3   " << setw(6) << m.tn << "   " << setw(6) << m.fp << endl;
        cout << "  Real 8   " << setw(6) << m.fn << "   " << setw(6) << m.tp << endl;
        cout << string(52, '-') << endl;
        cout << "  Tiempo entrenamiento: " << setprecision(4)
             << train_time_sec << " segundos" << endl;
        cout << string(52, '=') << endl;
    }
};

// ============================================================
// CLASE: Benchmarker
// Mide tiempo de CPU con alta precisión
// ============================================================
class Benchmarker {
private:
    time_point<high_resolution_clock> start_;

public:
    void start()  { start_ = high_resolution_clock::now(); }

    double elapsedSeconds() const {
        auto end = high_resolution_clock::now();
        return duration<double>(end - start_).count();
    }
};

// ============================================================
// FUNCIÓN DEMO: Genera datos sintéticos si no hay CSV
// Útil para pruebas en Qt Creator sin dataset descargado
// ============================================================
void generateSyntheticMNIST(Matrix_f& X, Vector_i& y, int n_samples = 2000) {
    mt19937 rng(42);
    normal_distribution<float> noise(0.0f, 30.0f);

    X.resize(n_samples, 784);
    y.resize(n_samples);

    // Plantilla simplificada de dígito 3 (curva derecha)
    VectorXf template3 = VectorXf::Zero(784);
    for (int r = 5; r < 23; r++) {
        for (int c = 5; c < 22; c++) {
            float dist_top = sqrt(pow(r - 9, 2) + pow(c - 14, 2));
            float dist_bot = sqrt(pow(r - 19, 2) + pow(c - 14, 2));
            float dist_mid = abs(r - 14) + abs(c - 10);
            if (dist_top < 5 || dist_bot < 5 || dist_mid < 3)
                template3(r * 28 + c) = 200.0f;
        }
    }

    // Plantilla simplificada de dígito 8 (dos óvalos)
    VectorXf template8 = VectorXf::Zero(784);
    for (int r = 3; r < 28; r++) {
        for (int c = 5; c < 24; c++) {
            float dist_top = sqrt(pow(r - 9,  2) + pow(c - 14, 2));
            float dist_bot = sqrt(pow(r - 19, 2) + pow(c - 14, 2));
            if ((dist_top > 4 && dist_top < 7) || (dist_bot > 4 && dist_bot < 7))
                template8(r * 28 + c) = 220.0f;
        }
    }

    for (int i = 0; i < n_samples; i++) {
        y(i) = i % 2;
        VectorXf base = (y(i) == 0) ? template3 : template8;
        for (int j = 0; j < 784; j++) {
            X(i, j) = max(0.0f, min(255.0f, base(j) + noise(rng)));
        }
    }

    cout << "[INFO] Dataset sintético generado: " << n_samples
         << " muestras (3 vs 8 simulados)" << endl;
}

// ============================================================
// MAIN
// ============================================================
int main(int argc, char* argv[]) {
    cout << "\n";
    cout << "============================================================\n";
    cout << "  PROYECTO FINAL — REDES NEURONALES\n";
    cout << "  Regresión Logística en C++ con Eigen\n";
    cout << "  Universidad Sergio Arboleda\n";
    cout << "============================================================\n\n";

    // ---- Cargar datos ----
    Matrix_f X;
    Vector_i y;

    string csv_path = (argc > 1) ? argv[1] : "mnist_binary_3_8.csv";
    cout << "[1/5] Cargando dataset..." << endl;

    bool loaded = DataLoader::loadCSV(csv_path, X, y);
    if (!loaded) {
        cout << "[INFO] CSV no encontrado. Usando datos sintéticos para demo.\n";
        generateSyntheticMNIST(X, y, 2000);
    }

    cout << "      Muestras: " << X.rows() << " | Características: " << X.cols() << endl;
    cout << "      Clase 0 (Dígito 3): " << (y.array() == 0).count() << endl;
    cout << "      Clase 1 (Dígito 8): " << (y.array() == 1).count() << "\n\n";

    // ---- Split ----
    cout << "[2/5] Dividiendo datos (80/20 estratificado)..." << endl;
    Matrix_f X_train, X_test;
    Vector_i y_train, y_test;
    DataLoader::trainTestSplit(X, y, X_train, y_train, X_test, y_test, 0.2f, 42);
    cout << "      Train: " << X_train.rows() << " | Test: " << X_test.rows() << "\n\n";

    // ---- Normalización (StandardScaler) ----
    cout << "[3/5] Normalizando datos (StandardScaler)..." << endl;
    Vector_f mean_v, std_v;
    DataLoader::standardScale(X_train, X_test, mean_v, std_v);
    cout << "      Normalización aplicada: media≈0, std≈1\n\n";

    // ---- Entrenamiento ----
    cout << "[4/5] Entrenando Regresión Logística con Eigen...\n";
    cout << "      Hiperparámetros: lr=0.1, epochs=500, lambda=0.001\n";
    cout << string(50, '-') << endl;

    LogisticRegressionEigen model(0.1f, 500, 0.001f);

    Benchmarker bench;
    bench.start();
    model.fit(X_train, y_train);
    double train_time = bench.elapsedSeconds();

    // ---- Evaluación ----
    cout << "\n[5/5] Evaluando modelo en conjunto de prueba...\n";
    Vector_i y_pred = model.predict(X_test);
    Evaluator::Metrics metrics = Evaluator::evaluate(y_test, y_pred);
    Evaluator::printReport(metrics, train_time);

    // ---- Exportar resultados ----
    ofstream results("resultados_cpp.txt");
    results << fixed << setprecision(6);
    results << "accuracy,"   << metrics.accuracy   << "\n";
    results << "precision,"  << metrics.precision  << "\n";
    results << "recall,"     << metrics.recall     << "\n";
    results << "f1_score,"   << metrics.f1_score   << "\n";
    results << "train_time," << train_time          << "\n";
    results << "tp,"         << metrics.tp          << "\n";
    results << "tn,"         << metrics.tn          << "\n";
    results << "fp,"         << metrics.fp          << "\n";
    results << "fn,"         << metrics.fn          << "\n";
    results.close();
    cout << "\n[OK] Resultados exportados a: resultados_cpp.txt\n";

    // ---- Exportar pesos (para visualización en Python) ----
    ofstream weights_file("pesos_cpp.csv");
    const Vector_f& w = model.getWeights();
    for (int i = 0; i < w.size(); i++) {
        weights_file << w(i);
        if (i < w.size() - 1) weights_file << ",";
    }
    weights_file << "\n" << model.getBias() << "\n";
    weights_file.close();
    cout << "[OK] Pesos exportados a: pesos_cpp.csv\n";

    cout << "\n✅ Ejecución completada exitosamente.\n\n";
    return 0;
}
