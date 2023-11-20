function limparNaN(obj) {
    for (var prop in obj) {
        if (obj.hasOwnProperty(prop) && typeof obj[prop] === 'number' && isNaN(obj[prop])) {
            obj[prop] = null; // ou substitua por outro valor padrão
        }
    }
}

function pesquisar(senadorCodigo) {
    console.log("Entrei no pesquisar");
    var query = document.getElementById('search-input').value;
    var url = `/senador/${senadorCodigo}/pesquisar/${encodeURIComponent(query)}`;

    // Realiza a solicitação AJAX para o Flask
    // Realiza a solicitação AJAX para o Flask
    fetch(url)
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP error! Status: ${response.status}`);
            }
            console.log("Response:"+response)
            return response.json();
        })
        .then(data => exibirResultados(data, senadorCodigo))
        .catch(error => console.error('Erro na solicitação AJAX:', error));
    
    event.preventDefault(); 
}



function exibirResultados(resultados, senadorCodigo) {
    

    var resultadosContainer = document.getElementById('resultados-container');
    resultadosContainer.innerHTML = ''; // Limpa qualquer conteúdo existente

    if (resultados.length === 0) {
        resultadosContainer.innerHTML = '<p>Nenhum resultado encontrado.</p>';
    } else {
        resultadosContainer.innerHTML = '<h3>Resultados da Pesquisa:</h3>';

        

        
        // Crie elementos HTML para cada resultado
        resultados.resultados.forEach(function (resultado) {
            

            var card = document.createElement('div');
            card.className = 'card my-3';

            var cardBody = document.createElement('div');
            cardBody.className = 'card-body';

            // Verificar se as propriedades existem antes de acessá-las
            if (resultado['Identificacao'] !== null && resultado['Identificacao'] !== undefined) {
                var titulo = document.createElement('h5');
                titulo.className = 'card-title text-primary';
                titulo.textContent = resultado['Identificacao'];
                cardBody.appendChild(titulo);
            }

            if (resultado['Ementa'] !== null && resultado['Ementa'] !== undefined) {
                var ementa = document.createElement('p');
                ementa.className = 'card-text';
                ementa.textContent = resultado['Ementa'];
                cardBody.appendChild(ementa);
            }

            card.appendChild(cardBody);            
            resultadosContainer.appendChild(card);

            
            
        });
        // Redirecionar para a página de resultados (substitua 'resultados_pesquisa' com o caminho real)
        window.location.href = '/senador/'+senadorCodigo+'/resultados_pesquisa/';
        
    }
}
