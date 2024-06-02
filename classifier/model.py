from torch.nn import Module, Linear, Dropout, Identity, Sigmoid, LayerNorm, PReLU
from torch import sigmoid, load 
from torch import no_grad, LongTensor, FloatTensor
from transformers import BertModel, BertTokenizer
import numpy as np
import os




class LinearBlock(Module):
    def __init__(self, input_size: int, output_size: int, batchnorm: bool = False, activation: bool = False, dropout: float = 0):
        super().__init__()
        self.fc = Linear(input_size, output_size)
        self.batchnorm = LayerNorm(output_size) if batchnorm else Identity()
        self.activation = PReLU(output_size) if activation else Identity()
        self.dropout = Dropout(dropout) if dropout > 0 else Identity()


    def forward(self, x: FloatTensor) -> FloatTensor:
        x = self.fc(x)
        x = self.batchnorm(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x




class BertClassifier(Module):
    def __init__(self, hidden_size: int = 1024, dropout: float = 0, output_size: int = 14, activation: bool = False, batchnorm: bool = True):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.fc1 = LinearBlock(768, hidden_size, batchnorm=batchnorm, activation=True, dropout=dropout)
        self.fc2 = LinearBlock(hidden_size, output_size, batchnorm=False, activation=False, dropout=0)
        self.activation = Sigmoid() if activation else Identity()


    def forward(self, input_ids: LongTensor, attention_mask: LongTensor, token_type_ids: LongTensor) -> FloatTensor:
        x = self.bert(input_ids, attention_mask, token_type_ids).last_hidden_state[:, 0, :]
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.activation(x)
        return x
    

    @no_grad()
    def predict(self, text: str, activation_threshold: float = 0.5) -> list[str]:
        self.eval()

        # Get BERT Inputs
        tokenizer: BertTokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        bert_inputs: dict[str, LongTensor] = tokenizer(text, return_tensors='pt', truncation=True, padding='max_length', max_length=512)

        # Prepare Classifier Inputs
        input_ids: LongTensor = bert_inputs['input_ids']
        attention_mask: LongTensor = bert_inputs['attention_mask']
        token_type_ids: LongTensor = bert_inputs['token_type_ids']

        # Prepare Classifier Output
        out: FloatTensor = self(input_ids, attention_mask, token_type_ids).squeeze()
        out: np.ndarray = sigmoid(out).cpu().numpy()
        out = np.uint8(out > activation_threshold)

        # Get Meshroot Labels
        meshroot_labels: list[str] = self.get_labels_from_prediction(out)

        return meshroot_labels
    

    @staticmethod
    def get_labels_from_prediction(prediciton: np.ndarray) -> list[str]:
        MESHROOT: list[str] = ['Chemicals and Drugs [D]', 'Organisms [B]', 'Analytical, Diagnostic and Therapeutic Techniques, and Equipment [E]', 
                                'Disciplines and Occupations [H]', 'Diseases [C]', 'Named Groups [M]', 'Psychiatry and Psychology [F]', 'Phenomena and Processes [G]', 
                                'Health Care [N]', 'Geographicals [Z]', 'Anthropology, Education, Sociology, and Social Phenomena [I]', 
                                'Technology, Industry, and Agriculture [J]', 'Information Science [L]', 'Anatomy [A]', 'Humanities [K]']
        
        LABEL_ENCODING: list[str] = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'L', 'M', 'N', 'Z']

        meshroot_labels: list[str] = []

        for i in range(len(prediciton)):
            if prediciton[i] == 1:
                for label in MESHROOT:
                    if f'[{LABEL_ENCODING[i]}]' in label:
                        meshroot_labels.append(label)
        return meshroot_labels


    

if __name__ == "__main__":
    model = BertClassifier(hidden_size=1024, dropout=0, output_size=14, activation=False)

    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'trained_models', 'Classifier_32BS_1024HS_2e-05LR_1717333910', 'Classifier_32BS_1024HS_2e-05LR.pt')

    model.load_state_dict(load(path), strict=True)

    text = r'Solitary lymphocytoma is a rare cutaneous manifestation of Lyme borreliosis that has been reported almost exclusively from Europe. This suggests that its etiologic \
        agent may be absent or extremely rare on the North American continent. All three species of B. burgdorferi sensu lato known to be associated with human Lyme borreliosis \
        (B. burgdorferi sensu stricto, B. garinii, and B. afzelii have been isolated in Europe, whereas only B. burgdorferi sensu stricto has been found \
        in North America. This suggests that either B. garinii or B. afzelii might be the etiologic agent of borrelial lymphocytoma. To investigate this hypothesis \
        we characterized five strains of B. burgdorferi sensu lato isolated from lymphocytoma lesions of patients residing in Slovenia. The methods used included: \
        large restriction fragment pattern analysis of restriction enzyme MluI-digested genomic DNA, plasmid profiling, protein profiling, ribotyping using 5S, \
        16S, and 23S rDNA probes, and polymerase chain reaction amplification of the rrf (5S)-rrl (23S) intergenic spacer region. Molecular subtyping showed that four of the \
        five isolates belonged to the species B. afzelii; however, this species is the predominant patient isolate in Slovenia and, therefore, may not represent a preferential \
        association with lymphocytoma. The fifth isolate appeared to be most closely related to the DN127 genomic group of organisms. \
        Further characterization of the isolate revealed that it possessed a unique molecular "fingerprint." The results not only show that borrelial lymphocytoma \
        can be caused by B. afzelii but also demonstrate an association with another genomic group of B. burgdorferi sensu lato that is present in North America as well.'

    labels = model.predict(text)  

    print(labels)