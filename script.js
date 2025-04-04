// Constantes et variables globales
const CANVAS_PADDING = 30;
let perceptron;
let trainingData = [];
let costHistory = [];
let isTraining = false;
let animationId;
let currentIteration = 0;

// Initialisation au chargement de la page
document.addEventListener('DOMContentLoaded', function() {
    // Initialiser les éléments de l'interface
    initializeUI();
    
    // Créer le perceptron
    resetPerceptron();
    
    // Générer des données d'exemple
    generateTrainingData();
    
    // Dessiner les visualisations initiales
    drawNetwork();
    drawDataPoints();
    drawCostFunction();
    drawGradientVisualization();
    
    // Ajouter les écouteurs d'événements
    setupEventListeners();
});

// Classe Perceptron
class Perceptron {
    constructor(inputSize = 2) {
        this.inputSize = inputSize;
        this.weights = Array(inputSize).fill().map(() => Math.random() * 2 - 1); // Poids aléatoires entre -1 et 1
        this.bias = Math.random() * 2 - 1; // Biais aléatoire entre -1 et 1
        this.activationFunction = 'sigmoid';
        this.learningRate = 0.1;
    }
    
    // Fonction de prédiction
    predict(inputs) {
        if (inputs.length !== this.inputSize) {
            throw new Error(`Le nombre d'entrées (${inputs.length}) ne correspond pas à la taille d'entrée du perceptron (${this.inputSize})`);
        }
        
        // Calcul de la somme pondérée
        let sum = this.bias;
        for (let i = 0; i < this.inputSize; i++) {
            sum += this.weights[i] * inputs[i];
        }
        
        // Application de la fonction d'activation
        return this.activate(sum);
    }
    
    // Fonctions d'activation
    activate(x) {
        switch (this.activationFunction) {
            case 'step':
                return x >= 0 ? 1 : 0;
            case 'sigmoid':
                return 1 / (1 + Math.exp(-x));
            case 'tanh':
                return Math.tanh(x);
            case 'relu':
                return Math.max(0, x);
            default:
                return 1 / (1 + Math.exp(-x)); // Sigmoid par défaut
        }
    }
    
    // Dérivée de la fonction d'activation
    activationDerivative(x) {
        switch (this.activationFunction) {
            case 'step':
                return 0; // La fonction step n'est pas dérivable, on utilise 0 par convention
            case 'sigmoid':
                const sigmoid = this.activate(x);
                return sigmoid * (1 - sigmoid);
            case 'tanh':
                const tanh = Math.tanh(x);
                return 1 - tanh * tanh;
            case 'relu':
                return x > 0 ? 1 : 0;
            default:
                const defaultSigmoid = this.activate(x);
                return defaultSigmoid * (1 - defaultSigmoid);
        }
    }
    
    // Fonction d'entraînement pour un exemple
    train(inputs, target) {
        // Calcul de la prédiction
        const prediction = this.predict(inputs);
        
        // Calcul de l'erreur
        const error = target - prediction;
        
        // Calcul de la somme pondérée pour obtenir la dérivée de la fonction d'activation
        let sum = this.bias;
        for (let i = 0; i < this.inputSize; i++) {
            sum += this.weights[i] * inputs[i];
        }
        
        // Calcul du gradient et mise à jour des poids
        const gradient = error * this.activationDerivative(sum);
        
        // Mise à jour des poids
        for (let i = 0; i < this.inputSize; i++) {
            this.weights[i] += this.learningRate * gradient * inputs[i];
        }
        
        // Mise à jour du biais
        this.bias += this.learningRate * gradient;
        
        // Retourner l'erreur au carré pour la fonction de coût
        return error * error;
    }
    
    // Fonction d'entraînement sur un ensemble de données
    trainBatch(data) {
        let totalCost = 0;
        
        for (const example of data) {
            const inputs = example.slice(0, this.inputSize);
            const target = example[this.inputSize];
            
            // Entraîner sur cet exemple et accumuler le coût
            totalCost += this.train(inputs, target);
        }
        
        // Retourner le coût moyen
        return totalCost / data.length;
    }
    
    // Fonction pour obtenir l'équation de la droite de séparation (pour 2D)
    getDecisionBoundary() {
        if (this.inputSize !== 2) {
            throw new Error("La frontière de décision ne peut être calculée que pour un perceptron à 2 entrées");
        }
        
        // Pour une fonction d'activation à seuil, la frontière est définie par w1*x1 + w2*x2 + b = 0
        // On peut réécrire cela comme x2 = (-w1*x1 - b) / w2
        
        // Si w2 est proche de zéro, la frontière est verticale
        if (Math.abs(this.weights[1]) < 1e-10) {
            return {
                isVertical: true,
                x: -this.bias / this.weights[0]
            };
        }
        
        return {
            isVertical: false,
            slope: -this.weights[0] / this.weights[1],
            intercept: -this.bias / this.weights[1]
        };
    }
}

// Fonctions d'initialisation
function initializeUI() {
    // Initialiser les valeurs affichées pour les contrôles
    document.getElementById('learning-rate-value').textContent = document.getElementById('learning-rate').value;
    document.getElementById('iterations-value').textContent = document.getElementById('iterations').value;
}

function resetPerceptron() {
    // Créer un nouveau perceptron
    perceptron = new Perceptron(2); // 2 entrées pour une visualisation 2D
    
    // Mettre à jour le taux d'apprentissage depuis le contrôle
    perceptron.learningRate = parseFloat(document.getElementById('learning-rate').value);
    
    // Mettre à jour la fonction d'activation depuis le contrôle
    perceptron.activationFunction = document.getElementById('activation-function').value;
    
    // Réinitialiser l'historique des coûts
    costHistory = [];
    currentIteration = 0;
    
    // Arrêter l'animation si elle est en cours
    if (isTraining) {
        isTraining = false;
        cancelAnimationFrame(animationId);
    }
}

function generateTrainingData() {
    // Vider les données existantes
    trainingData = [];
    
    // Générer des données pour deux classes
    // Classe 0: points en bas à gauche
    for (let i = 0; i < 50; i++) {
        const x1 = Math.random() * 0.5;
        const x2 = Math.random() * 0.5;
        trainingData.push([x1, x2, 0]);
    }
    
    // Classe 1: points en haut à droite
    for (let i = 0; i < 50; i++) {
        const x1 = 0.5 + Math.random() * 0.5;
        const x2 = 0.5 + Math.random() * 0.5;
        trainingData.push([x1, x2, 1]);
    }
    
    // Ajouter quelques points de bruit
    for (let i = 0; i < 10; i++) {
        const x1 = Math.random();
        const x2 = Math.random();
        // Classe opposée à celle attendue pour cette position
        const label = (x1 + x2 > 1) ? 0 : 1;
        trainingData.push([x1, x2, label]);
    }
    
    // Mélanger les données
    trainingData.sort(() => Math.random() - 0.5);
}

// Fonctions de dessin
function drawNetwork() {
    const canvas = document.getElementById('network-canvas');
    const ctx = canvas.getContext('2d');
    
    // Effacer le canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // Dimensions
    const width = canvas.width;
    const height = canvas.height;
    const centerX = width / 2;
    const centerY = height / 2;
    
    // Dessiner le fond
    ctx.fillStyle = '#f8f9fa';
    ctx.fillRect(0, 0, width, height);
    
    // Dessiner les nœuds d'entrée
    const inputNodeRadius = 25;
    const inputNodeY = height / 3;
    const inputNodeX1 = width / 3;
    const inputNodeX2 = 2 * width / 3;
    
    // Nœud d'entrée 1
    ctx.beginPath();
    ctx.arc(inputNodeX1, inputNodeY, inputNodeRadius, 0, 2 * Math.PI);
    ctx.fillStyle = '#17a2b8'; // Couleur info
    ctx.fill();
    ctx.strokeStyle = 'white';
    ctx.lineWidth = 2;
    ctx.stroke();
    
    // Texte pour le nœud d'entrée 1
    ctx.fillStyle = 'white';
    ctx.font = '16px Arial';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText('x₁', inputNodeX1, inputNodeY);
    
    // Nœud d'entrée 2
    ctx.beginPath();
    ctx.arc(inputNodeX2, inputNodeY, inputNodeRadius, 0, 2 * Math.PI);
    ctx.fillStyle = '#17a2b8'; // Couleur info
    ctx.fill();
    ctx.strokeStyle = 'white';
    ctx.lineWidth = 2;
    ctx.stroke();
    
    // Texte pour le nœud d'entrée 2
    ctx.fillStyle = 'white';
    ctx.font = '16px Arial';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText('x₂', inputNodeX2, inputNodeY);
    
    // Dessiner le nœud de sortie
    const outputNodeRadius = 30;
    const outputNodeY = 2 * height / 3;
    const outputNodeX = centerX;
    
    ctx.beginPath();
    ctx.arc(outputNodeX, outputNodeY, outputNodeRadius, 0, 2 * Math.PI);
    ctx.fillStyle = '#ff6b6b'; // Couleur accent
    ctx.fill();
    ctx.strokeStyle = 'white';
    ctx.lineWidth = 2;
    ctx.stroke();
    
    // Texte pour le nœud de sortie
    ctx.fillStyle = 'white';
    ctx.font = '16px Arial';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText('y', outputNodeX, outputNodeY);
    
    // Dessiner les connexions
    // Connexion 1 -> sortie
    const weight1 = perceptron.weights[0];
    const weight1Color = weight1 >= 0 ? '#28a745' : '#ffc107'; // Vert pour positif, jaune pour négatif
    const weight1Width = Math.min(8, Math.max(1, Math.abs(weight1) * 5)); // Épaisseur proportionnelle au poids
    
    ctx.beginPath();
    ctx.moveTo(inputNodeX1, inputNodeY + inputNodeRadius);
    ctx.lineTo(outputNodeX - outputNodeRadius * Math.cos(Math.PI/4), outputNodeY - outputNodeRadius * Math.sin(Math.PI/4));
    ctx.strokeStyle = weight1Color;
    ctx.lineWidth = weight1Width;
    ctx.stroke();
    
    // Texte pour le poids 1
    ctx.fillStyle = '#495057';
    ctx.font = '14px Arial';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    const w1Text = `w₁ = ${weight1.toFixed(2)}`;
    const w1X = (inputNodeX1 + outputNodeX - outputNodeRadius * Math.cos(Math.PI/4)) / 2;
    const w1Y = (inputNodeY + inputNodeRadius + outputNodeY - outputNodeRadius * Math.sin(Math.PI/4)) / 2;
    ctx.fillText(w1Text, w1X, w1Y);
    
    // Connexion 2 -> sortie
    const weight2 = perceptron.weights[1];
    const weight2Color = weight2 >= 0 ? '#28a745' : '#ffc107'; // Vert pour positif, jaune pour négatif
    const weight2Width = Math.min(8, Math.max(1, Math.abs(weight2) * 5)); // Épaisseur proportionnelle au poids
    
    ctx.beginPath();
    ctx.moveTo(inputNodeX2, inputNodeY + inputNodeRadius);
    ctx.lineTo(outputNodeX + outputNodeRadius * Math.cos(Math.PI/4), outputNodeY - outputNodeRadius * Math.sin(Math.PI/4));
    ctx.strokeStyle = weight2Color;
    ctx.lineWidth = weight2Width;
    ctx.stroke();
    
    // Texte pour le poids 2
    ctx.fillStyle = '#495057';
    ctx.font = '14px Arial';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    const w2Text = `w₂ = ${weight2.toFixed(2)}`;
    const w2X = (inputNodeX2 + outputNodeX + outputNodeRadius * Math.cos(Math.PI/4)) / 2;
    const w2Y = (inputNodeY + inputNodeRadius + outputNodeY - outputNodeRadius * Math.sin(Math.PI/4)) / 2;
    ctx.fillText(w2Text, w2X, w2Y);
    
    // Dessiner le biais
    ctx.fillStyle = '#495057';
    ctx.font = '14px Arial';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    const biasText = `b = ${perceptron.bias.toFixed(2)}`;
    ctx.fillText(biasText, outputNodeX, outputNodeY + outputNodeRadius + 20);
    
    // Dessiner la fonction d'activation
    ctx.fillStyle = '#495057';
    ctx.font = '14px Arial';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    let activationText;
    switch (perceptron.activationFunction) {
        case 'step':
            activationText = 'f(x) = step';
            break;
        case 'sigmoid':
            activationText = 'f(x) = sigmoid';
            break;
        case 'tanh':
            activationText = 'f(x) = tanh';
            break;
        case 'relu':
            activationText = 'f(x) = ReLU';
            break;
        default:
            activationText = 'f(x) = sigmoid';
    }
    ctx.fillText(activationText, outputNodeX, outputNodeY + outputNodeRadius + 40);
}

function drawDataPoints() {
    const canvas = document.getElementById('data-canvas');
    const ctx = canvas.getContext('2d');
    
    // Effacer le canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // Dimensions
    const width = canvas.width;
    const height = canvas.height;
    
    // Dessiner le fond
    ctx.fillStyle = '#f8f9fa';
    ctx.fillRect(0, 0, width, height);
    
    // Dessiner les axes
    ctx.strokeStyle = '#495057';
    ctx.lineWidth = 1;
    
    // Axe X
    ctx.beginPath();
    ctx.moveTo(CANVAS_PADDING, height - CANVAS_PADDING);
    ctx.lineTo(width - CANVAS_PADDING, height - CANVAS_PADDING);
    ctx.stroke();
    
    // Axe Y
    ctx.beginPath();
    ctx.moveTo(CANVAS_PADDING, height - CANVAS_PADDING);
    ctx.lineTo(CANVAS_PADDING, CANVAS_PADDING);
    ctx.stroke();
    
    // Étiquettes des axes
    ctx.fillStyle = '#495057';
    ctx.font = '12px Arial';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    
    // Étiquette X
    ctx.fillText('x₁', width - CANVAS_PADDING + 15, height - CANVAS_PADDING);
    
    // Étiquette Y
    ctx.fillText('x₂', CANVAS_PADDING, CANVAS_PADDING - 15);
    
    // Échelle
    const scaleX = (width - 2 * CANVAS_PADDING) / 1.0;
    const scaleY = (height - 2 * CANVAS_PADDING) / 1.0;
    
    // Dessiner les points de données
    for (const data of trainingData) {
        const x = CANVAS_PADDING + data[0] * scaleX;
        const y = height - CANVAS_PADDING - data[1] * scaleY;
        const label = data[2];
        
        ctx.beginPath();
        ctx.arc(x, y, 5, 0, 2 * Math.PI);
        ctx.fillStyle = label === 1 ? '#28a745' : '#ffc107'; // Vert pour classe 1, jaune pour classe 0
        ctx.fill();
        ctx.strokeStyle = '#495057';
        ctx.lineWidth = 1;
        ctx.stroke();
    }
    
    // Dessiner la frontière de décision si le perceptron est initialisé
    if (perceptron) {
        try {
            const boundary = perceptron.getDecisionBoundary();
            
            if (boundary.isVertical) {
                const x = CANVAS_PADDING + boundary.x * scaleX;
                
                ctx.beginPath();
                ctx.moveTo(x, CANVAS_PADDING);
                ctx.lineTo(x, height - CANVAS_PADDING);
                ctx.strokeStyle = '#ff6b6b';
                ctx.lineWidth = 2;
                ctx.stroke();
            } else {
                // Dessiner la ligne de x=0 à x=1
                const x1 = CANVAS_PADDING;
                const y1 = height - CANVAS_PADDING - (boundary.intercept) * scaleY;
                const x2 = CANVAS_PADDING + scaleX;
                const y2 = height - CANVAS_PADDING - (boundary.slope + boundary.intercept) * scaleY;
                
                ctx.beginPath();
                ctx.moveTo(x1, y1);
                ctx.lineTo(x2, y2);
                ctx.strokeStyle = '#ff6b6b';
                ctx.lineWidth = 2;
                ctx.stroke();
            }
        } catch (error) {
            console.error("Erreur lors du dessin de la frontière de décision:", error);
        }
    }
}

function drawCostFunction() {
    const canvas = document.getElementById('cost-canvas');
    const ctx = canvas.getContext('2d');
    
    // Effacer le canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // Dimensions
    const width = canvas.width;
    const height = canvas.height;
    
    // Dessiner le fond
    ctx.fillStyle = '#f8f9fa';
    ctx.fillRect(0, 0, width, height);
    
    // Dessiner les axes
    ctx.strokeStyle = '#495057';
    ctx.lineWidth = 1;
    
    // Axe X (itérations)
    ctx.beginPath();
    ctx.moveTo(CANVAS_PADDING, height - CANVAS_PADDING);
    ctx.lineTo(width - CANVAS_PADDING, height - CANVAS_PADDING);
    ctx.stroke();
    
    // Axe Y (coût)
    ctx.beginPath();
    ctx.moveTo(CANVAS_PADDING, height - CANVAS_PADDING);
    ctx.lineTo(CANVAS_PADDING, CANVAS_PADDING);
    ctx.stroke();
    
    // Étiquettes des axes
    ctx.fillStyle = '#495057';
    ctx.font = '12px Arial';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    
    // Étiquette X
    ctx.fillText('Itérations', width / 2, height - CANVAS_PADDING + 15);
    
    // Étiquette Y
    ctx.fillText('Coût', CANVAS_PADDING - 15, height / 2);
    
    // Si nous avons un historique de coûts, le dessiner
    if (costHistory.length > 0) {
        // Trouver le coût maximum pour l'échelle
        const maxCost = Math.max(...costHistory);
        const minCost = Math.min(...costHistory);
        const costRange = maxCost - minCost;
        
        // Échelles
        const scaleX = (width - 2 * CANVAS_PADDING) / (costHistory.length - 1 || 1);
        const scaleY = (height - 2 * CANVAS_PADDING) / (costRange || 1);
        
        // Dessiner la courbe de coût
        ctx.beginPath();
        ctx.moveTo(CANVAS_PADDING, height - CANVAS_PADDING - (costHistory[0] - minCost) * scaleY);
        
        for (let i = 1; i < costHistory.length; i++) {
            const x = CANVAS_PADDING + i * scaleX;
            const y = height - CANVAS_PADDING - (costHistory[i] - minCost) * scaleY;
            ctx.lineTo(x, y);
        }
        
        ctx.strokeStyle = '#ff6b6b';
        ctx.lineWidth = 2;
        ctx.stroke();
        
        // Afficher le coût actuel
        const currentCost = costHistory[costHistory.length - 1];
        ctx.fillStyle = '#495057';
        ctx.font = '14px Arial';
        ctx.textAlign = 'left';
        ctx.textBaseline = 'top';
        ctx.fillText(`Coût actuel: ${currentCost.toFixed(4)}`, CANVAS_PADDING + 10, CANVAS_PADDING + 10);
    } else {
        // Afficher un message si aucun entraînement n'a été effectué
        ctx.fillStyle = '#495057';
        ctx.font = '14px Arial';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText('Aucune donnée de coût disponible. Lancez l\'entraînement.', width / 2, height / 2);
    }
}

function drawGradientVisualization() {
    const canvas = document.getElementById('gradient-canvas');
    const ctx = canvas.getContext('2d');
    
    // Effacer le canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // Dimensions
    const width = canvas.width;
    const height = canvas.height;
    
    // Dessiner le fond
    ctx.fillStyle = '#f8f9fa';
    ctx.fillRect(0, 0, width, height);
    
    // Si nous n'avons pas encore d'historique de coûts, afficher un message
    if (costHistory.length === 0) {
        ctx.fillStyle = '#495057';
        ctx.font = '14px Arial';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText('Aucune donnée de gradient disponible. Lancez l\'entraînement.', width / 2, height / 2);
        return;
    }
    
    // Dessiner une surface 3D simplifiée représentant la fonction de coût
    // Nous allons dessiner une ellipse pour représenter les contours de la fonction de coût
    
    const centerX = width / 2;
    const centerY = height / 2;
    const radiusX = width / 3;
    const radiusY = height / 3;
    
    // Dessiner plusieurs ellipses concentriques pour représenter les contours
    for (let i = 5; i > 0; i--) {
        const ratio = i / 5;
        ctx.beginPath();
        ctx.ellipse(centerX, centerY, radiusX * ratio, radiusY * ratio, 0, 0, 2 * Math.PI);
        ctx.fillStyle = `rgba(255, 107, 107, ${0.1 + 0.1 * (5 - i)})`; // Plus foncé vers le centre
        ctx.fill();
        ctx.strokeStyle = '#ff6b6b';
        ctx.lineWidth = 1;
        ctx.stroke();
    }
    
    // Dessiner le point représentant la position actuelle dans l'espace des poids
    // Nous utilisons les deux premiers poids comme coordonnées (simplification)
    const w1Normalized = (perceptron.weights[0] + 1) / 2; // Normaliser entre 0 et 1
    const w2Normalized = (perceptron.weights[1] + 1) / 2; // Normaliser entre 0 et 1
    
    const pointX = centerX + (w1Normalized - 0.5) * 2 * radiusX;
    const pointY = centerY + (w2Normalized - 0.5) * 2 * radiusY;
    
    ctx.beginPath();
    ctx.arc(pointX, pointY, 8, 0, 2 * Math.PI);
    ctx.fillStyle = '#4a6fa5';
    ctx.fill();
    ctx.strokeStyle = 'white';
    ctx.lineWidth = 2;
    ctx.stroke();
    
    // Dessiner une flèche représentant la direction du gradient
    if (costHistory.length >= 2) {
        // Calculer une direction approximative du gradient basée sur la variation des poids
        // Dans un cas réel, nous calculerions le gradient analytiquement
        const arrowLength = 30;
        const arrowAngle = Math.atan2(perceptron.weights[1], perceptron.weights[0]);
        
        const arrowEndX = pointX - arrowLength * Math.cos(arrowAngle);
        const arrowEndY = pointY - arrowLength * Math.sin(arrowAngle);
        
        // Ligne principale de la flèche
        ctx.beginPath();
        ctx.moveTo(pointX, pointY);
        ctx.lineTo(arrowEndX, arrowEndY);
        ctx.strokeStyle = '#4a6fa5';
        ctx.lineWidth = 3;
        ctx.stroke();
        
        // Pointe de la flèche
        const arrowHeadLength = 10;
        const arrowHeadAngle = Math.PI / 6; // 30 degrés
        
        const arrowHead1X = arrowEndX + arrowHeadLength * Math.cos(arrowAngle - Math.PI + arrowHeadAngle);
        const arrowHead1Y = arrowEndY + arrowHeadLength * Math.sin(arrowAngle - Math.PI + arrowHeadAngle);
        
        const arrowHead2X = arrowEndX + arrowHeadLength * Math.cos(arrowAngle - Math.PI - arrowHeadAngle);
        const arrowHead2Y = arrowEndY + arrowHeadLength * Math.sin(arrowAngle - Math.PI - arrowHeadAngle);
        
        ctx.beginPath();
        ctx.moveTo(arrowEndX, arrowEndY);
        ctx.lineTo(arrowHead1X, arrowHead1Y);
        ctx.lineTo(arrowHead2X, arrowHead2Y);
        ctx.closePath();
        ctx.fillStyle = '#4a6fa5';
        ctx.fill();
    }
    
    // Légende
    ctx.fillStyle = '#495057';
    ctx.font = '14px Arial';
    ctx.textAlign = 'left';
    ctx.textBaseline = 'top';
    ctx.fillText('Espace des poids (w₁, w₂)', CANVAS_PADDING + 10, CANVAS_PADDING + 10);
    ctx.fillText('Point bleu: position actuelle', CANVAS_PADDING + 10, CANVAS_PADDING + 30);
    ctx.fillText('Flèche: direction du gradient', CANVAS_PADDING + 10, CANVAS_PADDING + 50);
    ctx.fillText('Contours: fonction de coût', CANVAS_PADDING + 10, CANVAS_PADDING + 70);
}

// Fonctions d'entraînement
function trainPerceptron() {
    if (isTraining) {
        return; // Déjà en cours d'entraînement
    }
    
    // Mettre à jour les paramètres du perceptron
    perceptron.learningRate = parseFloat(document.getElementById('learning-rate').value);
    perceptron.activationFunction = document.getElementById('activation-function').value;
    
    // Nombre total d'itérations
    const totalIterations = parseInt(document.getElementById('iterations').value);
    
    // Réinitialiser l'historique des coûts si nous recommençons
    if (currentIteration === 0) {
        costHistory = [];
    }
    
    // Démarrer l'entraînement
    isTraining = true;
    
    // Fonction d'animation pour l'entraînement
    function animate() {
        if (!isTraining || currentIteration >= totalIterations) {
            isTraining = false;
            return;
        }
        
        // Entraîner sur un lot de données
        const cost = perceptron.trainBatch(trainingData);
        costHistory.push(cost);
        
        // Mettre à jour les visualisations
        drawNetwork();
        drawDataPoints();
        drawCostFunction();
        drawGradientVisualization();
        
        // Incrémenter l'itération
        currentIteration++;
        
        // Continuer l'animation
        animationId = requestAnimationFrame(animate);
    }
    
    // Démarrer l'animation
    animate();
}

// Configuration des écouteurs d'événements
function setupEventListeners() {
    // Écouteur pour le bouton d'entraînement
    document.getElementById('train-button').addEventListener('click', trainPerceptron);
    
    // Écouteur pour le bouton de réinitialisation
    document.getElementById('reset-button').addEventListener('click', function() {
        resetPerceptron();
        drawNetwork();
        drawDataPoints();
        drawCostFunction();
        drawGradientVisualization();
    });
    
    // Écouteurs pour les contrôles de paramètres
    document.getElementById('learning-rate').addEventListener('input', function() {
        const value = this.value;
        document.getElementById('learning-rate-value').textContent = value;
        perceptron.learningRate = parseFloat(value);
    });
    
    document.getElementById('iterations').addEventListener('input', function() {
        const value = this.value;
        document.getElementById('iterations-value').textContent = value;
    });
    
    document.getElementById('activation-function').addEventListener('change', function() {
        perceptron.activationFunction = this.value;
        drawNetwork();
    });
}