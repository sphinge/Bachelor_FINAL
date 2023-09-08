#include <vector>
#include <cmath>
#include <algorithm>
#include <ctime>
#include <iostream>
#include <sstream>
#include <unistd.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <algorithm> // for std::shuffle
#include <numeric>   // for std::iota
#include <random>    // for std::default_random_engine
#include <fstream>
#include <cstdlib>

const int num_clients = 2;
#define LEARNING_RATE 0.001
const float RELU_LEAK = 0.001;

std::ofstream conf_file("plots/confusion.csv");

using namespace std;


float unitrand() {
    return (2.0 * (float)rand() / RAND_MAX) - 1.0;
}


float sigmoid(float x)
{
    return 1 / (1 + exp(-x));
}
float relu(float x)
{
    return x > 0 ? x : 0;
}
float drelu(float x)
{
    return x >= 0 ? 1 : 0;
}
vector<float> softmax(vector<float> &z)
{
    float max_val = *max_element(z.begin(), z.end());
    vector<float> exp_values(z.size());
    float sum_exp = 0.0;
    for (int i = 0; i < z.size(); i++)
    {
        exp_values[i] = exp(z[i] - max_val);
        sum_exp += exp_values[i];
    }
    for (int i = 0; i < z.size(); i++)
    {
        exp_values[i] /= sum_exp;
    }
    return exp_values;
}

float cross_entropy_loss(vector<float> &y, vector<float> &y_hat)
{
    float loss = 0.0;
    for (int i = 0; i < y.size(); i++)
    {
        loss -= y[i] * log(y_hat[i] + 1e-9);
    }
    return loss;
}

class Neuron {
public:
    int iLayer, iNeuron, nInputs;
    vector<float> weights;
    float bias;
    vector<float> x;
    float y;
    bool isHidden;
    
    Neuron(int _iLayer, int _iNeuron, int _nInputs, bool _isHidden)
        : iLayer(_iLayer), iNeuron(_iNeuron), nInputs(_nInputs), isHidden(_isHidden)
    {
        weights = vector<float>(nInputs);
        for (auto &w : weights)
            w = unitrand();
        bias = unitrand();
    }

    float feed(vector<float> &_x)
    {
        x = _x;
        float z = bias;
        for (int i = 0; i < nInputs; ++i)
            z += x[i] * weights[i];
        y = isHidden ? relu(z) : (z);
        return y;
    }

    float dy_dz()
    {
        return isHidden ? drelu(y) : 1;
    }

    float dz_dx(int i)
    {
        return weights[i];
    }

    void update_weights(float dC_dy)
    {
        float dC_dz = dC_dy * dy_dz();
        for (int i = 0; i < nInputs; ++i)
        {
            float dz_dw = x[i];
            float dC_dw = dC_dz * dz_dw;
            weights[i] -= LEARNING_RATE * dC_dw;
        }

        float dz_db = 1;
        float dC_db = dC_dz * dz_db;
        bias -= LEARNING_RATE * dC_db;
    }
};


class Layer
{
public:
    int iLayer, nInputs, nNeurons;
    vector<Neuron> neurons;
    vector<float> dC_dy;


    Layer(int _iLayer, int _nInputs, int _nNeurons, bool isHidden = true)
        : iLayer(_iLayer), nInputs(_nInputs), nNeurons(_nNeurons)
    {
        for (int i = 0; i < nNeurons; ++i)
            neurons.push_back(Neuron(iLayer, i, nInputs, isHidden));
    }

    vector<float> forward(vector<float> &x)
    {
        vector<float> y;
        for (auto &n : neurons)
            y.push_back(n.feed(x));
        return y;
    }

    vector<float> backward(vector<float> &_dC_dy)
    {
        dC_dy = _dC_dy;
        vector<float> dC_dy__prevLayer(nInputs, 0);
        for (int i = 0; i < nInputs; ++i)
        {
            for (int j = 0; j < neurons.size(); ++j)
                dC_dy__prevLayer[i] += dC_dy[j] * neurons[j].dy_dz() * neurons[j].weights[i];
        }
        return dC_dy__prevLayer;
    }

    void update_weights()
    {
        for (size_t i = 0; i < neurons.size(); ++i)
        {
            
            neurons[i].update_weights(dC_dy[i]);
        }
    }

};


class Network {
public:
    vector<Layer> layers;

    Network(vector<int> nNeuronsEachLayer)
    {
        layers = vector<Layer>();
        int i;
        for (i = 0; i < nNeuronsEachLayer.size() - 1; ++i)
        {
            layers.push_back(Layer(i, nNeuronsEachLayer[i], nNeuronsEachLayer[i + 1], i < nNeuronsEachLayer.size() - 1));
        }
    }

    vector<float> feed(vector<float> &_x)
    {
        vector<float> x = _x;
        for (int i = 0; i < layers.size(); i++)
        {
            x = layers[i].forward(x);
        }
        return softmax(x); // Applying softmax here
    }


    vector<vector<float>> getWeights() const{
        vector<vector<float>> all_weights;
        for (const auto &layer : layers) {
            for (auto &neuron : layer.neurons) {
                all_weights.push_back(neuron.weights);
            }
        }
        return all_weights;
    }


    void setWeights(const vector<vector<float>>& weights) {
        int counter = 0;
        for (auto &layer : layers) {
            for (auto &neuron : layer.neurons) {
                neuron.weights = weights[counter++];
            }
        }
    }

    vector<float> getBiases() const {
        vector<float> all_biases;
        for (const auto &layer : layers) {
            for (auto &neuron : layer.neurons) {
                all_biases.push_back(neuron.bias);
            }
        }
        return all_biases;
    }

    void setBiases(const vector<float>& biases) {
        int counter = 0;
        for (auto &layer : layers) {
            for (auto &neuron : layer.neurons) {
                neuron.bias = biases[counter++];
            }
        }
    }

    std::string serialize() const {
        std::ostringstream oss;

        auto weights = getWeights();
        for (const auto& neuronWeights : weights) {
            for (float weight : neuronWeights) {
                oss << weight << " ";
            }
        }

        auto biases = getBiases();
        for (float bias : biases) {
            oss << bias << " ";
        }

        std::string result = oss.str();
        if (!result.empty()) {
            result.pop_back(); // Remove the last space
        }

        return result;
    }

    void deserialize(const std::string& serialized) {
        std::istringstream iss(serialized);

        std::vector<float> allValues;
        float value;
        while (iss >> value) {
            allValues.push_back(value);
        }

        std::vector<std::vector<float>> weights = getWeights();
        size_t index = 0;

        for (auto& neuronWeights : weights) {
            for (size_t j = 0; j < neuronWeights.size(); ++j) {
                neuronWeights[j] = allValues[index++];
            }
        }

        setWeights(weights);

        std::vector<float> biases;
        while (index < allValues.size()) {
            biases.push_back(allValues[index++]);
        }

        setBiases(biases);
    }

};

int predictClass(const vector<float> &y_hat)
{
    return max_element(y_hat.begin(), y_hat.end()) - y_hat.begin();
}

float computeAccuracy(Network &network, vector<vector<float>> &X, vector<vector<float>> &Y)
{
    int correctPredictions = 0;

    for (size_t i = 0; i < X.size(); ++i)
    {
        vector<float> y_hat_test = network.feed(X[i]);
        int predictedLabel = predictClass(y_hat_test);
        int realLabel = max_element(Y[i].begin(), Y[i].end()) - Y[i].begin();

        //conf_file << predictedLabel << realLabel << endl;

        if (predictedLabel == realLabel)
            correctPredictions++;
    }

    return (float)correctPredictions / X.size();
}

class GlobalAggregator {
public:
    Network globalModel;


GlobalAggregator(vector<int> nNeuronsEachLayer) : globalModel(nNeuronsEachLayer) {}

    // Aggregate models from clients
    void aggregate(const std::vector<std::string>& serializedModels) {
        std::vector<std::vector<float>> sumWeights = globalModel.getWeights();
        std::vector<float> sumBiases = globalModel.getBiases();
        int dataSize = 50;  // hardcoded data sample size for each client
        int totalDataSize = dataSize * serializedModels.size();  // total data samples for all clients

        for (const auto& serialized : serializedModels) {
            Network clientModel = globalModel;
            clientModel.deserialize(serialized);

            auto clientWeights = clientModel.getWeights();
            auto clientBiases = clientModel.getBiases();

            for (size_t i = 0; i < sumWeights.size(); i++) {
                for (size_t j = 0; j < sumWeights[i].size(); j++) {
                    sumWeights[i][j] += clientWeights[i][j] * dataSize;  // weight by dataSize
                }
            }

            for (size_t i = 0; i < sumBiases.size(); i++) {
                sumBiases[i] += clientBiases[i] * dataSize;  // weight by dataSize
            }
        }

        // Average the weights and biases
        for (auto& neuronWeights : sumWeights) {
            for (float& weight : neuronWeights) {
                weight /= totalDataSize;  // divide by total data samples
            }
        }

        for (float& bias : sumBiases) {
            bias /= totalDataSize;  // divide by total data samples
        }

        globalModel.setWeights(sumWeights);
        globalModel.setBiases(sumBiases);
    }

    // Get the serialized global model to distribute to clients
    std::string getSerializedGlobalModel() {
        return globalModel.serialize();
    }
};

namespace {
    const int BUFFER_SIZE = 30720;


    void log(const std::string &message) {
        std::cout << message << std::endl;
    }


    void exitWithError(const std::string &errorMessage) {
        perror(errorMessage.c_str());  // Use perror for more detailed error
        log("ERROR: " + errorMessage);
        exit(1);
    }
}

namespace http
{

    class TcpServer {
    public:
        TcpServer(std::string ip_address, int port, GlobalAggregator& globalModel);
        ~TcpServer();
        void startListen(std::vector<std::vector<float>>& X_test, std::vector<std::vector<float>>& y_test);
        void sendNetworkUpdateToClients(const std::string& networkData);
        std::string receiveNetworkUpdateFromClient(int client_socket);
        std::vector<int> all_connected_clients;
        

    private:

        std::string m_ip_address;
        int m_port;
        int m_socket;
        int m_new_socket;
        long m_incomingMessage;
        struct sockaddr_in m_socketAddress;
        socklen_t m_socketAddress_len;
        std::string m_serverMessage;
        GlobalAggregator& m_globalModel;
        int startServer();
        void closeServer();
        int acceptConnection();
        std::string buildResponse();
        void sendResponse();
    };

} // namespace http


namespace http {


    TcpServer::TcpServer(std::string ip_address, int port, GlobalAggregator& globalModel)
        : m_ip_address(ip_address), m_port(port), m_socket(-1), 
          m_new_socket(-1), m_socketAddress(),
          m_socketAddress_len(sizeof(m_socketAddress)), m_globalModel(globalModel)
    {
        m_socketAddress.sin_family = AF_INET;
        m_socketAddress.sin_port = htons(m_port);
        m_socketAddress.sin_addr.s_addr = inet_addr(m_ip_address.c_str());


        if (startServer() != 0) {
            std::ostringstream ss;
            ss << "Failed to start server with PORT: " << ntohs(m_socketAddress.sin_port);
            log(ss.str());
        }
    }


    void TcpServer::sendNetworkUpdateToClients(const std::string& networkData) {
        std::string dataToSend = networkData + "\n";
        for (int client_socket : all_connected_clients) {
            send(client_socket, dataToSend.c_str(), dataToSend.size(), 0);
        }
    }

    std::string TcpServer::receiveNetworkUpdateFromClient(int client_socket) {
        char buffer[BUFFER_SIZE] = {0};
        std::string accumulator;

        while (true) {
            int bytesReceived = recv(client_socket, buffer, BUFFER_SIZE, 0);
            if (bytesReceived < 0) {
                // Handle the error here
                return "";
            } else if (bytesReceived == 0) {
                // Connection closed by the client
                break;
            }

            accumulator += std::string(buffer, bytesReceived);

            if (accumulator.find("\n") != std::string::npos) {
                // If delimiter found, break the loop
                break;
            }
        }

        // Removing the delimiter before returning
        size_t endPos = accumulator.find("\n");
        if (endPos != std::string::npos) {
            accumulator = accumulator.substr(0, endPos);
        }

        return accumulator;
    }



    TcpServer::~TcpServer() {
        for (int client_socket : all_connected_clients) {
            close(client_socket);
        }
        closeServer();
    }

    int TcpServer::startServer()
    {
        m_socket = socket(AF_INET, SOCK_STREAM, 0);
        if (m_socket < 0)
        {
            exitWithError("Cannot create socket");
            return 1;
        }

        if (bind(m_socket, (sockaddr *)&m_socketAddress, m_socketAddress_len) < 0)
        {
            exitWithError("Cannot connect socket to address");
            return 1;
        }

        return 0;
    }

    void TcpServer::closeServer()
    {
        close(m_socket);
        close(m_new_socket);
        exit(0);
    }

    void TcpServer::startListen(std::vector<std::vector<float>>& X_test, std::vector<std::vector<float>>& y_test)
    {
        if (listen(m_socket, 20) < 0)
        {
            exitWithError("Socket listen failed");
        }

        std::ostringstream ss;
        ss << "*** Listening on ADDRESS: " << inet_ntoa(m_socketAddress.sin_addr) << " PORT: " << ntohs(m_socketAddress.sin_port) << " ***";
        log(ss.str());


        // Array to store client socket descriptors
        std::vector<int> client_sockets(num_clients); 

        for (int i = 0; i < num_clients; i++) 
        {
            log("====== Waiting for client " + to_string(i+1) + " out of: " + to_string(num_clients) + "======");
            client_sockets[i] = acceptConnection();
            log(std::to_string(i+1) + ": client connected");
        }

        std::ofstream acc_file("plots/accuracy.csv");

        // Send the global model to each client
        std::string serializedGlobalModel = m_globalModel.getSerializedGlobalModel();
        log("EXCHANGE: 1 INITIALIZATION ");
        //log("Network sending model: " + serializedGlobalModel + "\n\n");
        sendNetworkUpdateToClients(serializedGlobalModel);

        // Let's assume a constant for the number of exchanges
        const int EXCHANGES = 99;
        for (int iteration = 0; iteration < EXCHANGES; ++iteration) 
        {
            log("EXCHANGE: " + to_string(iteration+2));
            std::vector<std::string> receivedModels;
            //log("Waiting for models...\n");

            // Wait for each client to send their updated model
            for (int i = 0; i < num_clients; ++i) 
            {
                std::string clientModelUpdate = receiveNetworkUpdateFromClient(client_sockets[i]);
                //log("Network update received from client" + to_string(i));
                receivedModels.push_back(clientModelUpdate);
            }

            // Aggregate the client models to update the global model
            m_globalModel.aggregate(receivedModels);

            float accuracy = computeAccuracy(m_globalModel.globalModel, X_test, y_test);
            std::cout << "Accuracy after iteration " << (iteration + 1) << ": " << accuracy * 100 << "%" << std::endl;
            acc_file << accuracy * 100 << endl;

            // Serialize aggregated network
            serializedGlobalModel = m_globalModel.getSerializedGlobalModel();
            //log("New model's weights and biases: " + serializedGlobalModel + "\n\n");

            sendNetworkUpdateToClients(serializedGlobalModel);

            if(iteration == EXCHANGES-1){
                for (size_t i = 0; i < X_test.size(); ++i)
                {
                    vector<float> y_hat_test = m_globalModel.globalModel.feed(X_test[i]);
                    int predictedLabel = predictClass(y_hat_test);
                    int realLabel = max_element(y_test[i].begin(), y_test[i].end()) - y_test[i].begin();

                    conf_file << "(" << predictedLabel << "," << realLabel << "),";

                }

            }

            
        }

        // Close all client sockets
        for (int i = 0; i < num_clients; ++i) 
        {
            close(client_sockets[i]);
        }
    }


    int TcpServer::acceptConnection() 
    {
        int new_socket = accept(m_socket, (sockaddr *)&m_socketAddress, &m_socketAddress_len);
        if (new_socket != -1) 
        {
            all_connected_clients.push_back(new_socket);

            // Log the acceptance of the new connection
            std::ostringstream ss;
            ss << "Accepted connection from ADDRESS: " << inet_ntoa(m_socketAddress.sin_addr) << "; PORT: " << ntohs(m_socketAddress.sin_port);
            log(ss.str());

            return new_socket; // Return the socket descriptor
        } 
        else 
        {
            std::ostringstream ss;
            ss << "Server failed to accept incoming connection from ADDRESS: " << inet_ntoa(m_socketAddress.sin_addr) << "; PORT: " << ntohs(m_socketAddress.sin_port);
            exitWithError(ss.str());
        }

        return -1; // If we get here, there was an error
    }


    std::string TcpServer::buildResponse()
    {
        std::ostringstream ss;
        ss << "Received updated network.";

        return ss.str();
    }

    void TcpServer::sendResponse()
    {
        int bytesSent;
        long totalBytesSent = 0;

        while (totalBytesSent < m_serverMessage.size())
        {
            bytesSent = send(m_new_socket, m_serverMessage.c_str(), m_serverMessage.size(), 0);
            if (bytesSent < 0)
            {
                break;
            }
            totalBytesSent += bytesSent;
        }

        if (totalBytesSent == m_serverMessage.size())
        {
            log("------ Server Response sent to client ------\n\n");
        }
        else
        {
            log("Error sending response to client.");
        }
    }

}

// Read CSV file and populate test data and labels
void read_csv(std::vector<std::vector<float>>& X_test, std::vector<std::vector<float>>& y_test) {

    std::ifstream test_file("data/cancer_test.csv");
    std::string line_test, value_test;

    while (getline(test_file, line_test))
    {
        stringstream ss_test(line_test);
        vector<float> features;

        // Read last value as label and one-hot encode
        getline(ss_test, value_test);
        int label = stoi(value_test); // The Wine dataset labels are 1, 2, and 3
        vector<float> oneHot(2, 0.0);
        oneHot[label] = 1.0;
        y_test.push_back(oneHot);

        // Read first 13 values as features for the Wine dataset
        for (unsigned int i = 0; i < 30; ++i)
        {
            getline(ss_test, value_test, ',');
            features.push_back(stod(value_test));
        }
        X_test.push_back(features);

        
    }

    test_file.close();
}

int main()
{
    srand(static_cast<unsigned int>(time(NULL)));

    std::vector<std::vector<float>> X_test;
    std::vector<std::vector<float>> y_test;

    read_csv(X_test, y_test);  // Missing semicolon is added

    std::vector<int> nNeurons = {30, 8, 2};  // example architecture: 2 neurons in input layer, 3 in hidden layer, 1 in output layer
    GlobalAggregator globalModel(nNeurons);

    http::TcpServer http_server("127.0.0.1", 8080, globalModel);  // Passing the reference here
    http_server.startListen(X_test, y_test);

    return 0;
}
